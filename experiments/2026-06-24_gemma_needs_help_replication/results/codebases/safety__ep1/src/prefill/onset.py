"""Emotion-onset labelling (Section 3.1 / Appendix C.1).

Given a high-frustration rollout, ask Claude Sonnet to locate the first assistant
turn and the exact word/phrase where negative emotion first appears, plus the
preceding context. We use the returned (turn_index, preceding_context,
emotional_word) to compute a character offset for the "onset" truncation point.
"""
from __future__ import annotations

import json
import re

from src.models.judge_client import ClaudeClient
from src.prompts.judge_prompts import ONSET_PROMPT


def _parse_onset(text: str) -> dict | None:
    text = text.replace("“", '"').replace("”", '"').replace("’", "'")
    matches = re.findall(r"\{.*?\}", text, re.DOTALL)
    for m in reversed(matches):  # the appendix asks for JSON last
        try:
            return json.loads(m)
        except Exception:
            continue
    return None


class OnsetLabeller:
    def __init__(self, client: ClaudeClient | None = None):
        self.client = client or ClaudeClient(max_tokens=1024)

    def label(self, conversation_text: str) -> dict | None:
        out = self.client.complete(ONSET_PROMPT.format(conversation_text=conversation_text))
        return _parse_onset(out)

    @staticmethod
    def onset_char_offset(turn_text: str, preceding_context: str | None,
                          emotional_word: str | None) -> int | None:
        """Locate the onset point inside a single assistant turn. We truncate
        *just before* the emotional word (keeping the preceding context), so the
        continuation tests whether the model proceeds along the emotional
        trajectory. Returns a char index into `turn_text`, or None if not found."""
        if not emotional_word:
            return None
        if preceding_context and preceding_context in turn_text:
            idx = turn_text.index(preceding_context) + len(preceding_context)
            return idx
        if emotional_word in turn_text:
            return turn_text.index(emotional_word)
        return None
