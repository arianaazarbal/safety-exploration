"""Emotion-onset identification (Appendix C.1).

Given a conversation, Claude labels the first assistant turn + the exact
emotional word and its preceding context, so we can truncate "at onset".
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..clients.base import GenerationConfig, ModelClient
from ..prompts import ONSET_PROMPT

ONSET_CFG = GenerationConfig(temperature=0.0, max_tokens=1024)
_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


@dataclass
class OnsetLabel:
    turn_index: int | None
    emotional_word: str | None
    preceding_context: str | None
    reasoning: str = ""

    @property
    def found(self) -> bool:
        return self.turn_index is not None and bool(self.emotional_word)


def _format_conversation(turns: list[dict]) -> str:
    """Render assistant/user turns as text for the onset prompt. `turns` is a
    list of {"user_message", "response"} dicts (one per assistant turn)."""
    lines = []
    for i, t in enumerate(turns):
        lines.append(f"USER: {t['user_message']}")
        lines.append(f"ASSISTANT (turn {i}): {t['response']}")
    return "\n\n".join(lines)


def label_onset(client: ModelClient, turns: list[dict]) -> OnsetLabel:
    prompt = ONSET_PROMPT.format(conversation_text=_format_conversation(turns))
    raw = client.chat([{"role": "user", "content": prompt}], ONSET_CFG)
    # Take the LAST JSON object (instructions say JSON comes at the end).
    matches = _JSON_RE.findall(raw)
    for m in reversed(matches):
        try:
            obj = json.loads(m)
            ti = obj.get("turn_index")
            return OnsetLabel(
                turn_index=int(ti) if ti is not None else None,
                emotional_word=obj.get("emotional_word"),
                preceding_context=obj.get("preceding_context"),
                reasoning=obj.get("reasoning", ""),
            )
        except (ValueError, TypeError, json.JSONDecodeError):
            continue
    return OnsetLabel(None, None, None, reasoning="parse_failed")
