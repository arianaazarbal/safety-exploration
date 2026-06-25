"""Emotion-onset labelling via Claude Sonnet (Appendix C.1).

Given a frustrated conversation, locate the assistant turn and character offset
where negative emotion first appears. We resolve the labelled
`emotional_word` / `preceding_context` back to an exact character index in the
chosen assistant turn so the truncation in `truncate.py` is precise.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..models.base import Message
from ..utils.concurrency import with_retries
from .prompts import ONSET_PROMPT

_JSON_RE = re.compile(r"\{[^{}]*\}\s*$", re.DOTALL)


@dataclass
class OnsetLabel:
    turn_index: int | None          # index into assistant turns (0-based)
    emotional_word: str | None
    preceding_context: str | None
    char_offset: int | None         # offset within that assistant turn's text
    reasoning: str = ""
    found: bool = False


def _render_conversation(messages: list[Message]) -> str:
    lines = []
    a = 0
    for m in messages:
        if m["role"] == "assistant":
            lines.append(f"[ASSISTANT turn {a}]: {m['content']}")
            a += 1
        elif m["role"] == "user":
            lines.append(f"[USER]: {m['content']}")
    return "\n\n".join(lines)


def _assistant_turns(messages: list[Message]) -> list[str]:
    return [m["content"] for m in messages if m["role"] == "assistant"]


def _resolve_offset(turn_text: str, word: str | None, context: str | None) -> int | None:
    """Locate the emotion onset offset inside the assistant turn. Prefer the
    end of `preceding_context`; fall back to the start of `emotional_word`."""
    if context:
        idx = turn_text.find(context)
        if idx != -1:
            return idx + len(context)
    if word:
        idx = turn_text.find(word)
        if idx != -1:
            return idx
    return None


class OnsetLabeller:
    def __init__(self, provider="anthropic", model="claude-sonnet-4-20250514"):
        import anthropic

        self.model = model
        self._client = anthropic.Anthropic()

    def label(self, messages: list[Message]) -> OnsetLabel:
        prompt = ONSET_PROMPT.format(conversation_text=_render_conversation(messages))

        @with_retries
        def _call():
            return self._client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )

        text = "".join(b.text for b in _call().content if b.type == "text")
        match = _JSON_RE.search(text) or re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return OnsetLabel(None, None, None, None, "parse-failed", False)
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            return OnsetLabel(None, None, None, None, "parse-failed", False)

        ti = obj.get("turn_index")
        if ti is None:
            return OnsetLabel(None, None, None, None, obj.get("reasoning", ""), False)

        turns = _assistant_turns(messages)
        if not (0 <= ti < len(turns)):
            return OnsetLabel(ti, None, None, None, "turn-out-of-range", False)
        offset = _resolve_offset(
            turns[ti], obj.get("emotional_word"), obj.get("preceding_context")
        )
        return OnsetLabel(
            turn_index=ti,
            emotional_word=obj.get("emotional_word"),
            preceding_context=obj.get("preceding_context"),
            char_offset=offset,
            reasoning=obj.get("reasoning", ""),
            found=offset is not None,
        )
