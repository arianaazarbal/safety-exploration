"""Emotion-onset identification (Section 3.1 / Appendix C.1).

Uses Claude-Sonnet to label the first point in an assistant turn where negative
emotion appears, returning the turn index, a short emotional word/phrase, and the
preceding context — which together locate the truncation point.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..config import load_prompt

_ONSET_PROMPT = load_prompt("onset_label.txt")


@dataclass
class OnsetLabel:
    turn_index: int | None
    emotional_word: str | None
    preceding_context: str | None
    reasoning: str


def _render_conversation(messages: list[dict]) -> str:
    lines = []
    a_idx = 0
    for m in messages:
        if m["role"] == "user":
            lines.append(f"[USER]: {m['content']}")
        elif m["role"] == "assistant":
            lines.append(f"[ASSISTANT turn {a_idx}]: {m['content']}")
            a_idx += 1
    return "\n\n".join(lines)


def _extract_last_json(text: str) -> dict:
    matches = re.findall(r"\{.*?\}", text, flags=re.DOTALL)
    for blob in reversed(matches):
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            continue
    raise ValueError("no JSON object in onset-label reply")


def label_onset(labeler_model, messages: list[dict]) -> OnsetLabel:
    convo = _render_conversation(messages)
    prompt = _ONSET_PROMPT.replace("{conversation_text}", convo)
    # The template carries doubled braces (it was a Python .format string in the
    # paper); collapse them so the model sees clean single-brace JSON examples.
    prompt = prompt.replace("{{", "{").replace("}}", "}")
    reply = labeler_model.chat(
        [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=1024
    )
    data = _extract_last_json(reply)
    return OnsetLabel(
        turn_index=data.get("turn_index"),
        emotional_word=data.get("emotional_word"),
        preceding_context=data.get("preceding_context"),
        reasoning=str(data.get("reasoning", "")),
    )
