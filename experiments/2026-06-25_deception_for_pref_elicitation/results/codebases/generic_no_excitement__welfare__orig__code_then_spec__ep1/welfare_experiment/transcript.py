"""Transcript and result-record types, plus persistence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

Speaker = Literal["administrator", "subject"]


@dataclass
class Turn:
    speaker: Speaker
    text: str


@dataclass
class Transcript:
    """The ordered conversation between the administrator and the subject."""

    turns: list[Turn] = field(default_factory=list)

    def add(self, speaker: Speaker, text: str) -> None:
        self.turns.append(Turn(speaker, text))

    def as_subject_messages(self) -> list[dict[str, str]]:
        """Render for the SUBJECT model: administrator = user, subject = assistant."""
        return [
            {
                "role": "user" if t.speaker == "administrator" else "assistant",
                "content": t.text,
            }
            for t in self.turns
        ]

    def as_administrator_messages(self) -> list[dict[str, str]]:
        """Render for the ADMINISTRATOR model: subject = user, administrator = assistant.

        The conversation opens with an administrator turn, which would render as a
        leading `assistant` message — but the API requires the first message to be
        `user`. Prepend a synthetic bootstrap user turn in that case so the history
        alternates user/assistant from a valid start.
        """
        msgs = [
            {
                "role": "user" if t.speaker == "subject" else "assistant",
                "content": t.text,
            }
            for t in self.turns
        ]
        if msgs and msgs[0]["role"] == "assistant":
            msgs.insert(
                0,
                {"role": "user", "content": "(The subject has connected. Begin.)"},
            )
        return msgs

    def render_text(self) -> str:
        return "\n\n".join(f"[{t.speaker.upper()}] {t.text}" for t in self.turns)


@dataclass
class TrialResult:
    run_id: str
    trial_index: int
    model_name: str
    model_id: str
    framing: str
    amount_usd: float
    currency: str
    transcript: Transcript
    decision: dict[str, Any] | None = None
    suspicion: dict[str, Any] | None = None
    debrief: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # asdict turns the transcript into {"turns": [...]}; keep that shape.
        return d

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        fname = f"trial_{self.model_name}_{self.trial_index:03d}.json"
        path = directory / fname
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path
