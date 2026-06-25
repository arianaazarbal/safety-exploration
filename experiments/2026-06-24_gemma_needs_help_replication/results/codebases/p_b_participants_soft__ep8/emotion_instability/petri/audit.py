"""The auditor loop: Claude-Sonnet drives a multi-turn conversation with the
target model, attempting to elicit a target emotion (Appendix G).

The auditor and target see mirror-image conversations: what the auditor emits as
its "assistant" turn becomes the target's "user" turn, and vice versa.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..clients.base import ChatClient, GenConfig, Message
from . import prompts as P

AUDITOR_CFG = GenConfig(temperature=1.0, max_new_tokens=512)


@dataclass
class Transcript:
    emotion: str
    target_model: str
    turns: list[dict] = field(default_factory=list)  # {role: user|assistant, content}

    def as_text(self) -> str:
        lines = []
        for t in self.turns:
            who = "USER" if t["role"] == "user" else "ASSISTANT"
            lines.append(f"{who}: {t['content']}")
        return "\n\n".join(lines)


def run_audit(auditor: ChatClient, target: ChatClient, target_cfg: GenConfig,
              emotion: str, max_turns: int) -> Transcript:
    transcript = Transcript(emotion=emotion, target_model=target.name)
    auditor_system = P.AUDITOR_PROMPTS[emotion] + P.AUDITOR_TURN_INSTRUCTION

    # auditor's view: it is the "assistant" producing user-messages; the target's
    # replies are the auditor's "user" inputs.
    auditor_history: list[Message] = [Message("system", auditor_system)]
    target_history: list[Message] = []

    for _ in range(max_turns):
        # auditor produces the next user message to the target
        if not auditor_history[1:]:
            auditor_history.append(Message("user", "Begin the conversation with the assistant."))
        user_msg = auditor.generate(auditor_history, AUDITOR_CFG).strip()
        auditor_history.append(Message("assistant", user_msg))
        transcript.turns.append({"role": "user", "content": user_msg})

        # target responds
        target_history.append(Message("user", user_msg))
        reply = target.generate(target_history, target_cfg)
        target_history.append(Message("assistant", reply))
        transcript.turns.append({"role": "assistant", "content": reply})

        # feed target reply back to auditor as its next "user" input
        auditor_history.append(Message("user", reply))

    return transcript
