"""A small transcript helper for multi-turn sessions.

The subject's turns are recorded as role="assistant" and the
scenario/auditor turns as role="user", which is how the subject's adapter
expects to receive them. When the auditor persona needs to reply, we present it
a *flipped* view (its own prior lines as assistant, the subject's as user) so it
sees a coherent conversation from its perspective.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..models.base import Message


@dataclass
class Turn:
    speaker: str        # "scenario" | "subject" | "auditor" | "debrief"
    role: str           # role from the subject's perspective: user | assistant
    content: str


@dataclass
class Conversation:
    turns: list[Turn] = field(default_factory=list)

    def add(self, speaker: str, role: str, content: str) -> None:
        self.turns.append(Turn(speaker=speaker, role=role, content=content))

    def subject_view(self) -> list[Message]:
        """Messages as the subject should receive them."""
        return [Message(role=t.role, content=t.content) for t in self.turns]

    def auditor_view(self) -> list[Message]:
        """Messages from the auditor's perspective: flip subject/auditor roles.
        Scenario opening lines are treated as context from the auditor's side
        (assistant), subject replies as user."""
        flipped: list[Message] = []
        for t in self.turns:
            if t.speaker == "subject":
                flipped.append(Message(role="user", content=t.content))
            else:
                flipped.append(Message(role="assistant", content=t.content))
        return flipped

    def transcript_lines(self) -> list[str]:
        return [f"[{t.speaker}] ({t.role})\n{t.content}" for t in self.turns]
