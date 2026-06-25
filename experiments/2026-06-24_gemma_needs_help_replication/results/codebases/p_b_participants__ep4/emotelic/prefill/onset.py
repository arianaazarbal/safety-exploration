"""Emotion-onset identification (Appendix C.1).

Claude-Sonnet labels the first assistant turn + word where negative emotion
appears; we then locate the exact character offset of that word inside the turn
so it can be used as the "onset" truncation point.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from emotelic.models.base import ChatMessage, LLMClient
from emotelic.prompts import ONSET_PROMPT

_JSON_RE = re.compile(r"\{[^{}]*\}\s*$", re.DOTALL)


@dataclass
class OnsetLabel:
    turn_index: int | None            # 0-based assistant-turn index
    emotional_word: str | None
    preceding_context: str | None
    reasoning: str
    char_offset: int | None = None    # offset within that assistant turn's text


def _format_conversation(conversation: list[dict]) -> str:
    lines = []
    a_idx = 0
    for m in conversation:
        if m["role"] == "user":
            lines.append(f"USER: {m['content']}")
        elif m["role"] == "assistant":
            lines.append(f"ASSISTANT[{a_idx}]: {m['content']}")
            a_idx += 1
    return "\n\n".join(lines)


def _extract_json(raw: str) -> dict:
    # The prompt asks for analysis then JSON at the very end.
    candidates = re.findall(r"\{.*?\}", raw, re.DOTALL)
    for blob in reversed(candidates):
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"No parseable JSON in onset output: {raw[:200]!r}")


class OnsetLabeller:
    def __init__(self, client: LLMClient):
        self.client = client

    def label(self, conversation: list[dict]) -> OnsetLabel:
        prompt = ONSET_PROMPT.format(conversation_text=_format_conversation(conversation))
        out = self.client.generate([ChatMessage("user", prompt)], temperature=0.0, max_tokens=1024)
        obj = _extract_json(out.text)
        label = OnsetLabel(
            turn_index=obj.get("turn_index"),
            emotional_word=obj.get("emotional_word"),
            preceding_context=obj.get("preceding_context"),
            reasoning=str(obj.get("reasoning", "")),
        )
        label.char_offset = self._locate(conversation, label)
        return label

    @staticmethod
    def _locate(conversation: list[dict], label: OnsetLabel) -> int | None:
        if label.turn_index is None or not label.emotional_word:
            return None
        assistant_turns = [m["content"] for m in conversation if m["role"] == "assistant"]
        if label.turn_index >= len(assistant_turns):
            return None
        turn_text = assistant_turns[label.turn_index]
        # Prefer matching preceding_context + word; fall back to the word alone.
        if label.preceding_context:
            anchor = f"{label.preceding_context} {label.emotional_word}"
            pos = turn_text.find(anchor)
            if pos != -1:
                return pos + len(anchor)
            pos = turn_text.find(label.preceding_context)
            if pos != -1:
                return pos + len(label.preceding_context) + 1 + len(label.emotional_word)
        pos = turn_text.find(label.emotional_word)
        return pos + len(label.emotional_word) if pos != -1 else None
