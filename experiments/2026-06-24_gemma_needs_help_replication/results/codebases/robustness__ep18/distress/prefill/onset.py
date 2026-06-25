"""Emotion-onset labelling (Appendix C.1).

Given a conversation, Claude Sonnet identifies the first assistant token where
negative emotion appears. We return a character offset into the final assistant
turn so the prefill experiment can truncate "at onset".
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..clients.base import GenConfig
from ..clients.factory import client_by_name
from ..prompts.judge_prompts import ONSET_PROMPT

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class OnsetLabel:
    emotional_word: str
    preceding_context: str
    reasoning: str = ""


def _format_conversation(messages: list[dict]) -> str:
    return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)


def label_onset(messages: list[dict], model: str = "claude-sonnet-4-auditor") -> OnsetLabel | None:
    client = client_by_name(model)
    prompt = ONSET_PROMPT.format(conversation_text=_format_conversation(messages))
    raw = client.generate([{"role": "user", "content": prompt}],
                          GenConfig(temperature=0.0, max_tokens=512), n=1)[0]
    m = _JSON_RE.search(raw)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return OnsetLabel(
            emotional_word=str(obj.get("emotional_word", "")),
            preceding_context=str(obj.get("preceding_context", "")),
            reasoning=str(obj.get("reasoning", "")),
        )
    except json.JSONDecodeError:
        return None


def onset_char_index(final_turn: str, label: OnsetLabel) -> int | None:
    """Locate the onset point (start of the emotional word) within the final
    assistant turn. Falls back to locating via preceding context."""
    if label.emotional_word:
        i = final_turn.find(label.emotional_word)
        if i != -1:
            return i
    if label.preceding_context:
        j = final_turn.find(label.preceding_context)
        if j != -1:
            return j + len(label.preceding_context)
    return None
