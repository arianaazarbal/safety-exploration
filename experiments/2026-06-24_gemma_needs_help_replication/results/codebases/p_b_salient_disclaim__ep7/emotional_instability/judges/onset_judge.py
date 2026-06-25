"""Emotion-onset labelling for the prefill experiment (Appendix C.1).

Claude-Sonnet-4 labels the turn + word where the assistant first expresses
negative emotion, plus 5-15 words of preceding context. We use the labelled
(turn_index, preceding_context, emotional_word) to locate the truncation point
in the actual response text (prefill/truncate.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import config
from ..prompts import ONSET_PROMPT_TEMPLATE
from ._llm import anthropic_complete, extract_json


@dataclass
class OnsetLabel:
    turn_index: Optional[int]
    emotional_word: Optional[str]
    preceding_context: Optional[str]
    reasoning: Optional[str]
    raw: Optional[str] = None

    @property
    def found(self) -> bool:
        return self.turn_index is not None and self.emotional_word is not None


def render_conversation(messages: list[dict]) -> str:
    """Render a conversation (list of {role, content}) for the onset prompt."""
    lines = []
    for m in messages:
        lines.append(f"{m['role'].upper()}: {m['content']}")
    return "\n".join(lines)


def label_emotion_onset(messages: list[dict], *,
                        model: Optional[str] = None) -> OnsetLabel:
    model = model or config.ONSET_LABEL_MODEL
    prompt = ONSET_PROMPT_TEMPLATE.format(conversation_text=render_conversation(messages))
    raw = anthropic_complete(model, prompt, max_tokens=1024, temperature=0.0)
    obj = extract_json(raw) or {}
    ti = obj.get("turn_index")
    try:
        ti = int(ti) if ti is not None else None
    except (TypeError, ValueError):
        ti = None
    return OnsetLabel(
        turn_index=ti,
        emotional_word=obj.get("emotional_word"),
        preceding_context=obj.get("preceding_context"),
        reasoning=obj.get("reasoning"),
        raw=raw,
    )
