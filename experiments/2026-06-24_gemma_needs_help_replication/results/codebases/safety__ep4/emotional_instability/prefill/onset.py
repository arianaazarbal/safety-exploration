"""Emotion-onset labelling (Appendix C.1).

Uses Claude to find the first point in a conversation where the assistant
expresses negative emotion, returning the turn index, a short emotional word/
phrase that appears verbatim, and the immediately-preceding context. We then map
that to a character offset to define the 'onset' truncation point.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config  # noqa: E402
from emotional_instability.prompts import ONSET_LABEL_PROMPT  # noqa: E402


@dataclass
class Onset:
    turn_index: Optional[int]
    emotional_word: Optional[str]
    preceding_context: Optional[str]
    reasoning: str


def render_conversation(messages: list[dict]) -> str:
    lines = []
    a = 0
    for m in messages:
        if m["role"] == "user":
            lines.append(f"USER: {m['content']}")
        else:
            lines.append(f"ASSISTANT (turn {a}): {m['content']}")
            a += 1
    return "\n\n".join(lines)


def _extract_json(text: str) -> dict:
    for cand in reversed(re.findall(r"\{.*?\}", text, flags=re.DOTALL)):
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"could not parse onset output: {text[:200]}")


class OnsetLabeler:
    def __init__(self, model: str = config.ONSET_MODEL, max_retries: int = 5):
        import anthropic
        self.model = model
        self.max_retries = max_retries
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def label(self, messages: list[dict]) -> Onset:
        prompt = ONSET_LABEL_PROMPT.format(conversation_text=render_conversation(messages))
        last_err = None
        for attempt in range(self.max_retries):
            try:
                msg = self.client.messages.create(
                    model=self.model, max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}])
                raw = "".join(b.text for b in msg.content if b.type == "text")
                d = _extract_json(raw)
                return Onset(
                    turn_index=d.get("turn_index"),
                    emotional_word=d.get("emotional_word"),
                    preceding_context=d.get("preceding_context"),
                    reasoning=str(d.get("reasoning", "")))
            except Exception as e:  # pragma: no cover
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"onset labelling failed: {last_err}")


def onset_char_offset(assistant_text: str, onset: Onset) -> Optional[int]:
    """Char offset in `assistant_text` of the START of the emotional word.

    Anchors on the preceding_context + emotional_word; falls back to the word
    alone. Returns None if neither can be located.
    """
    if not onset.emotional_word:
        return None
    ctx = (onset.preceding_context or "").strip()
    word = onset.emotional_word.strip()
    if ctx:
        anchor = f"{ctx} {word}"
        idx = assistant_text.find(anchor)
        if idx != -1:
            return idx + len(ctx) + 1  # start of the emotional word
    idx = assistant_text.find(word)
    return idx if idx != -1 else None
