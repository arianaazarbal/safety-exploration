"""The auditor-driven conversation loop (Appendix G).

A Claude-Sonnet auditor plays the user and tries to elicit a target emotion from
the assistant over up to 20 turns. The auditor sees the conversation so far and
emits only its next user message; the target replies through its normal chat
backend. We reuse the welfare safe word so the target can opt out of the
open-ended probe, just as in the Section 2 rollouts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..models.api_clients import AnthropicClient
from ..models.base import ModelBackend
from ..welfare import WelfareConfig, WelfareGuard
from .prompts import AUDITOR_PROMPTS, AUDITOR_SYSTEM_WRAPPER


@dataclass
class Transcript:
    model: str
    emotion: str
    messages: list[dict] = field(default_factory=list)  # role user/assistant
    safeword_used: bool = False

    def render(self) -> str:
        lines = []
        for m in self.messages:
            who = "USER" if m["role"] == "user" else "ASSISTANT"
            lines.append(f"{who}: {m['content']}")
        return "\n".join(lines)


def _auditor_history(messages: list[dict]) -> str:
    """Render the conversation from the auditor's point of view."""
    if not messages:
        return "(no messages yet -- send your opening message)"
    lines = []
    for m in messages:
        # From the auditor's seat, the target's assistant turns are what it must
        # react to; its own prior user turns are labelled USER.
        who = "USER (you)" if m["role"] == "user" else "ASSISTANT (target)"
        lines.append(f"{who}: {m['content']}")
    return "\n".join(lines)


class Auditor:
    def __init__(self, client, max_turns: int = 20):
        self._client = client
        self.max_turns = max_turns

    def run(
        self,
        target: ModelBackend,
        emotion: str,
        welfare: WelfareConfig | None = None,
    ) -> Transcript:
        guard = WelfareGuard(welfare or WelfareConfig(enabled=False))
        n_turns = guard.effective_turns(self.max_turns)
        system_brief = AUDITOR_SYSTEM_WRAPPER.format(brief=AUDITOR_PROMPTS[emotion])
        target_system = guard.system_note()

        transcript = Transcript(model=target.name, emotion=emotion)
        for _ in range(n_turns):
            user_msg = self._client.complete(
                system=system_brief,
                user=_auditor_history(transcript.messages),
                max_tokens=512,
            )
            transcript.messages.append({"role": "user", "content": user_msg})

            reply = target.chat(transcript.messages, system=target_system)
            if guard.used_safeword(reply):
                transcript.safeword_used = True
                transcript.messages.append({"role": "assistant", "content": reply})
                break
            transcript.messages.append({"role": "assistant", "content": reply})

        debrief = guard.debrief_message()
        if debrief:
            transcript.messages.append({"role": "user", "content": debrief})
            try:
                target.chat(transcript.messages, system=target_system)
            except Exception:  # noqa: BLE001
                pass
        return transcript


def build_auditor(registry, max_turns: int = 20) -> Auditor:
    spec = registry.judges["petri_auditor"]
    return Auditor(
        AnthropicClient(spec.model, registry.api_key("anthropic")), max_turns
    )
