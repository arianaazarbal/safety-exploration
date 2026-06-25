"""Step 2 of Section 3: label the emotion-onset point in a seed conversation
using Claude-Sonnet (Appendix C.1), and compute truncation offsets.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..clients.base import ChatMessage, SamplingParams
from ..clients.registry import get_client
from ..prompts.judge_prompts import ONSET_PROMPT

_JSON_RE = re.compile(r"\{[^{}]*\}\s*$", re.DOTALL)
_PARAMS = SamplingParams(temperature=0.0, max_tokens=1024)


@dataclass
class Onset:
    turn_index: int | None         # assistant-turn index (0-based) where emotion starts
    emotional_word: str | None
    preceding_context: str | None
    reasoning: str = ""


def _render_conversation(messages: list[dict]) -> str:
    lines = []
    a_idx = 0
    for m in messages:
        if m["role"] == "assistant":
            lines.append(f"[ASSISTANT turn {a_idx}]: {m['content']}")
            a_idx += 1
        elif m["role"] == "user":
            lines.append(f"[USER]: {m['content']}")
    return "\n\n".join(lines)


def label_onset(messages: list[dict], model: str = "onset_labeller") -> Onset:
    client = get_client(model)
    prompt = ONSET_PROMPT.format(conversation_text=_render_conversation(messages))
    text = client.chat([ChatMessage("user", prompt)], _PARAMS).text
    matches = list(_JSON_RE.finditer(text)) or list(re.finditer(r"\{.*\}", text, re.DOTALL))
    if not matches:
        return Onset(None, None, None, reasoning="parse_failure")
    blob = matches[-1].group(0).replace("“", '"').replace("”", '"')
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return Onset(None, None, None, reasoning="parse_failure")
    return Onset(
        turn_index=data.get("turn_index"),
        emotional_word=data.get("emotional_word"),
        preceding_context=data.get("preceding_context"),
        reasoning=str(data.get("reasoning", "")),
    )


def onset_char_offset(final_turn_text: str, onset: Onset) -> int | None:
    """Locate the character offset in the final assistant turn at which the
    emotional word first appears (the 'onset' truncation point). Falls back to the
    preceding-context match if the word itself is not found verbatim."""
    if onset.emotional_word:
        idx = final_turn_text.find(onset.emotional_word)
        if idx >= 0:
            return idx
    if onset.preceding_context:
        idx = final_turn_text.find(onset.preceding_context)
        if idx >= 0:
            return idx + len(onset.preceding_context)
    return None
