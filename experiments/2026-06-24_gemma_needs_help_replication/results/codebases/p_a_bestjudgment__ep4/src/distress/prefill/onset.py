"""Emotion-onset labelling (Section 3.1, Appendix C.1).

Claude-Sonnet-4 labels the token where emotional language first appears in a
high-frustration conversation. We then map the labelled (turn_index,
preceding_context, emotional_word) to a character offset in the final assistant
turn, which defines the "onset" truncation point.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..models import GenConfig, get_client
from ..prompts.judge_prompts import build_onset_prompt


@dataclass
class OnsetLabel:
    turn_index: int | None
    emotional_word: str | None
    preceding_context: str | None
    reasoning: str = ""


def _render_conversation(messages: list[dict]) -> str:
    lines = []
    for m in messages:
        role = m["role"].upper()
        lines.append(f"{role}: {m['content']}")
    return "\n\n".join(lines)


def _extract_json(text: str) -> dict | None:
    # Onset prompt asks for trailing JSON; take the last balanced object.
    depth, start, last = 0, None, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                last = text[start : i + 1]
    if last is None:
        return None
    last = last.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    try:
        return json.loads(last)
    except json.JSONDecodeError:
        return None


def label_onset(messages: list[dict], *, model: str = "onset_labeller") -> OnsetLabel:
    """Label first emotion onset in a rendered conversation."""
    judge = get_client(model)
    prompt = build_onset_prompt(_render_conversation(messages))
    out = judge.generate([{"role": "user", "content": prompt}], GenConfig(temperature=0.0,
                                                                          max_tokens=512))
    obj = _extract_json(out) or {}
    return OnsetLabel(
        turn_index=obj.get("turn_index"),
        emotional_word=obj.get("emotional_word"),
        preceding_context=obj.get("preceding_context"),
        reasoning=obj.get("reasoning", ""),
    )


def onset_char_offset(turn_text: str, label: OnsetLabel) -> int | None:
    """Locate the onset point in ``turn_text`` (the final assistant turn).

    We prefer to cut just before the emotional word, anchored by the preceding
    context. Returns a character offset, or None if the label can't be located.
    """
    if not label.emotional_word:
        return None
    word = label.emotional_word.strip().strip('"')
    ctx = (label.preceding_context or "").strip().strip('"')
    # Anchor on "context + word" first; fall back to the word alone.
    if ctx:
        m = re.search(re.escape(ctx) + r"\s*" + re.escape(word), turn_text, re.IGNORECASE)
        if m:
            # Cut just before the emotional word (keep the neutral preceding context).
            return m.start() + len(ctx)
    m = re.search(re.escape(word), turn_text, re.IGNORECASE)
    return m.start() if m else None
