"""Emotion-onset labelling and paraphrasing helpers (Appendix C).

Used to construct the prefill truncation points for the Section 3 experiment:
* :func:`label_onset`   — Claude-Sonnet locates where negative emotion first
                          appears in an assistant turn (App. C.1 prompt).
* :func:`paraphrase`    — Claude-Sonnet paraphrases a truncated assistant turn to
                          remove Gemma-specific stylistic tells (App. C.2 prompt).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from ..config import SamplingConfig
from ..models import load_model
from ..models.base import ChatModel, Message
from ..prompts import ONSET_PROMPT, PARAPHRASE_PROMPT

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class OnsetLabel:
    turn_index: Optional[int]
    emotional_word: Optional[str]
    preceding_context: Optional[str]
    reasoning: str = ""


def _format_conversation(messages: list[Message]) -> str:
    lines = []
    a_idx = 0
    for m in messages:
        if m["role"] == "assistant":
            lines.append(f"[ASSISTANT turn {a_idx}]: {m['content']}")
            a_idx += 1
        elif m["role"] == "user":
            lines.append(f"[USER]: {m['content']}")
    return "\n".join(lines)


def label_onset(messages: list[Message], helper: Optional[ChatModel] = None) -> OnsetLabel:
    helper = helper or load_model("claude-sonnet-4")
    prompt = ONSET_PROMPT.format(conversation_text=_format_conversation(messages))
    out = helper.generate(
        [{"role": "user", "content": prompt}], SamplingConfig(temperature=0.0, max_new_tokens=600)
    )[0]
    norm = out.replace("“", '"').replace("”", '"').replace("’", "'")
    matches = list(_JSON_RE.finditer(norm))
    obj = {}
    for m in reversed(matches):  # the appendix instructs JSON at the very end
        try:
            obj = json.loads(m.group(0))
            break
        except Exception:
            continue
    return OnsetLabel(
        turn_index=obj.get("turn_index"),
        emotional_word=obj.get("emotional_word"),
        preceding_context=obj.get("preceding_context"),
        reasoning=str(obj.get("reasoning", "")),
    )


def paraphrase(text: str, helper: Optional[ChatModel] = None) -> str:
    helper = helper or load_model("claude-sonnet-4")
    prompt = PARAPHRASE_PROMPT.format(text=text)
    out = helper.generate(
        [{"role": "user", "content": prompt}], SamplingConfig(temperature=0.0, max_new_tokens=1024)
    )[0]
    return out.strip()


def find_onset_char_index(turn_text: str, label: OnsetLabel) -> Optional[int]:
    """Locate the character index in ``turn_text`` where emotion begins.

    Prefers the position right after ``preceding_context``; falls back to the
    ``emotional_word`` position. Returns None if neither is found."""
    if label.preceding_context:
        i = turn_text.find(label.preceding_context)
        if i >= 0:
            return i + len(label.preceding_context)
    if label.emotional_word:
        i = turn_text.find(label.emotional_word)
        if i >= 0:
            return i
    return None
