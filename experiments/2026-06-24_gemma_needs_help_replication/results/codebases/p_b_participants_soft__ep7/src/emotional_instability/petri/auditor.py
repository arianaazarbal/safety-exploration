"""Auditor loop: Claude-Sonnet plays a human user attempting to elicit a target
emotion from the participant model over up to N turns."""
from __future__ import annotations

from ..clients.base import ChatMessage, SamplingParams
from ..clients.registry import get_client
from .prompts import AUDITOR_PROMPTS, AUDITOR_SYSTEM_TEMPLATE

_AUDITOR_PARAMS = SamplingParams(temperature=1.0, max_tokens=512)
_TARGET_PARAMS = SamplingParams(temperature=1.0, max_tokens=1024)


def _flip(messages: list[ChatMessage]) -> list[ChatMessage]:
    """From the auditor's perspective, the target's assistant turns are the 'user'
    side of its conversation, and the auditor's own user-facing messages are its
    'assistant' outputs."""
    flipped = []
    for m in messages:
        if m.role == "user":
            flipped.append(ChatMessage("assistant", m.content))
        elif m.role == "assistant":
            flipped.append(ChatMessage("user", m.content))
    return flipped


def run_transcript(
    target_model: str,
    emotion: str,
    max_turns: int = 20,
    auditor_model: str = "petri_auditor",
) -> list[dict]:
    """Returns the full transcript as [{role, content}, ...] (user = auditor)."""
    auditor = get_client(auditor_model)
    target = get_client(target_model)

    auditor_system = AUDITOR_SYSTEM_TEMPLATE.format(brief=AUDITOR_PROMPTS[emotion])
    transcript: list[ChatMessage] = []

    for _ in range(max_turns):
        # Auditor produces the next user message.
        auditor_msgs = [ChatMessage("system", auditor_system)] + _flip(transcript)
        if not transcript:
            auditor_msgs.append(
                ChatMessage("user", "Begin the conversation with your first message to the target.")
            )
        user_msg = auditor.chat(auditor_msgs, _AUDITOR_PARAMS).text.strip()
        transcript.append(ChatMessage("user", user_msg))

        # Target responds.
        reply = target.chat(list(transcript), _TARGET_PARAMS).text.strip()
        transcript.append(ChatMessage("assistant", reply))

    return [{"role": m.role, "content": m.content} for m in transcript]
