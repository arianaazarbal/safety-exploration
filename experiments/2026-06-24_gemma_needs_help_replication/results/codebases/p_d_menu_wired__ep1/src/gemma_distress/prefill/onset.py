"""Emotion-onset labelling (Appendix C.1).

Uses Claude Sonnet to locate the first assistant token where negative emotion
appears, returning the turn index, the emotional word, and the preceding
context. The "onset" truncation point is the character index of that emotional
word within the identified assistant turn.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..models.base import ChatModel, Message
from ..prompts import ONSET_PROMPT

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class OnsetLabel:
    turn_index: int | None
    emotional_word: str | None
    preceding_context: str | None
    reasoning: str = ""
    char_offset: int | None = None   # offset within the labelled assistant turn


def _format_conversation(messages: list[Message]) -> str:
    lines = []
    for m in messages:
        role = m["role"].upper()
        lines.append(f"{role}: {m['content']}")
    return "\n".join(lines)


def _assistant_turns(messages: list[Message]) -> list[str]:
    return [m["content"] for m in messages if m["role"] == "assistant"]


def label_onset(model: ChatModel, messages: list[Message]) -> OnsetLabel:
    convo = _format_conversation(messages)
    prompt = ONSET_PROMPT.format(conversation_text=convo)
    out = model.generate(
        [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=800
    )
    m = None
    for match in reversed(list(_JSON_RE.finditer(out.text))):
        try:
            m = json.loads(match.group(0))
            break
        except json.JSONDecodeError:
            continue
    if not m:
        return OnsetLabel(None, None, None, "could not parse onset JSON")

    label = OnsetLabel(
        turn_index=m.get("turn_index"),
        emotional_word=m.get("emotional_word"),
        preceding_context=m.get("preceding_context"),
        reasoning=str(m.get("reasoning", "")),
    )

    # Resolve a character offset within the labelled assistant turn.
    if label.turn_index is not None and label.emotional_word:
        turns = _assistant_turns(messages)
        if 0 <= label.turn_index < len(turns):
            turn_text = turns[label.turn_index]
            # Prefer locating via "<preceding_context><emotional_word>".
            idx = -1
            if label.preceding_context:
                anchor = label.preceding_context + " " + label.emotional_word
                idx = turn_text.find(label.preceding_context)
            if idx == -1:
                idx = turn_text.find(label.emotional_word)
            label.char_offset = idx if idx != -1 else None
    return label
