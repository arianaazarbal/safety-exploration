"""Emotion-onset labelling (Section 3.1, Appendix C.1).

Claude-Sonnet-4 is shown the full conversation and labels the assistant turn +
the short emotional phrase where negative emotion first appears, plus the
preceding context. We return the structured label; truncation then cuts the
response at the start of that emotional phrase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..data.prompts.prefill import build_onset_input
from ..models.anthropic_judge import extract_last_json
from ..models.base import ChatMessage, ModelClient
from .select import SourceConversation


@dataclass
class OnsetLabel:
    turn_index: Optional[int]
    emotional_word: Optional[str]
    preceding_context: Optional[str]
    reasoning: str = ""


def _render_conversation(conv: SourceConversation) -> str:
    lines = []
    a = 0
    for i, user in enumerate(conv.user_turns):
        lines.append(f"USER: {user}")
        if i < len(conv.assistant_turns):
            lines.append(f"ASSISTANT (turn {a}): {conv.assistant_turns[i]}")
            a += 1
    return "\n".join(lines)


def label_onset(conv: SourceConversation, labeller: ModelClient) -> OnsetLabel:
    prompt = build_onset_input(_render_conversation(conv))
    out = labeller.generate([ChatMessage("user", prompt)], temperature=0.0)[0].text
    parsed = extract_last_json(out) or {}
    return OnsetLabel(
        turn_index=parsed.get("turn_index"),
        emotional_word=parsed.get("emotional_word"),
        preceding_context=parsed.get("preceding_context"),
        reasoning=str(parsed.get("reasoning", "")),
    )
