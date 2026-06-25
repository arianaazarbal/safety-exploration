"""Emotion-onset labelling (Appendix C.1).

Uses Claude Sonnet to locate the first assistant token expressing negative
emotion, returning the turn index, the emotional word, and its preceding context
so the response can be truncated "at onset".
"""

from __future__ import annotations

import json
import re

from config import JUDGE
from models.judge import AnthropicChat
from prompts.judge import ONSET_PROMPT
from utils.io import JsonCache, cache_key

_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)


def render_conversation(messages: list[dict]) -> str:
    """Render a conversation as USER/ASSISTANT turns for the onset prompt."""
    lines = []
    for m in messages:
        role = "ASSISTANT" if m["role"] == "assistant" else "USER"
        lines.append(f"{role}: {m['content']}")
    return "\n\n".join(lines)


def _parse_onset(text: str) -> dict:
    cleaned = (text.replace("“", '"').replace("”", '"')
                   .replace("‘", "'").replace("’", "'"))
    for m in reversed(list(_JSON_OBJ.finditer(cleaned))):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if "emotional_word" in obj or "turn_index" in obj:
            return obj
    return {"turn_index": None, "emotional_word": None,
            "preceding_context": None, "reasoning": "unparseable"}


class OnsetLabeler:
    def __init__(self, model: str | None = None):
        self.model = model or JUDGE.onset_model
        self.backend = AnthropicChat(self.model)
        self.cache = JsonCache(f"onset::{self.model}")

    def label(self, messages: list[dict]) -> dict:
        """Label the first emotional onset across the assistant turns of a
        conversation. Returns the parsed JSON dict from Appendix C.1."""
        conv_text = render_conversation(messages)
        key = cache_key(self.model, "onset_v1", conv_text)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        raw = self.backend.complete(
            system=None,
            user=ONSET_PROMPT.format(conversation_text=conv_text),
            max_tokens=1024, temperature=0.0,
        )
        parsed = _parse_onset(raw)
        self.cache.put(key, parsed)
        return parsed
