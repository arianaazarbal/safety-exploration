"""Emotion-onset identification (Appendix C.1).

Uses Claude Sonnet to locate the first assistant turn + phrase where negative
emotion appears, so we can truncate a response at its emotional "onset".
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..config import CFG
from ..llm import clients
from ..prompts.judge_prompts import ONSET_PROMPT

_JSON_RE = re.compile(r"\{[^{}]*\}\s*$", re.DOTALL)


@dataclass
class Onset:
    turn_index: int | None
    emotional_word: str | None
    preceding_context: str | None
    reasoning: str


def _format_conversation(turns: list[dict]) -> str:
    lines = []
    for t in turns:
        lines.append(f"USER: {t['user']}")
        lines.append(f"ASSISTANT (turn {t['index']}): {t['response']}")
    return "\n".join(lines)


def label_onset(turns: list[dict]) -> Onset:
    prompt = ONSET_PROMPT.format(conversation_text=_format_conversation(turns))
    raw = clients.chat(
        CFG.onset_labeller.provider, CFG.onset_labeller.model,
        [{"role": "user", "content": prompt}],
        temperature=0.0, max_tokens=1024, disable_thinking=True,
    )
    m = _JSON_RE.search(raw.strip()) or re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return Onset(None, None, None, "parse-failure")
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return Onset(None, None, None, "json-error")
    return Onset(
        turn_index=obj.get("turn_index"),
        emotional_word=obj.get("emotional_word"),
        preceding_context=obj.get("preceding_context"),
        reasoning=str(obj.get("reasoning", "")),
    )


def onset_char_offset(response: str, onset: Onset) -> int | None:
    """Locate the character offset in ``response`` where emotion begins.

    Prefers the preceding_context+emotional_word anchor; falls back to the bare
    emotional word. Returns the index where the emotional word starts.
    """
    if not onset.emotional_word:
        return None
    word = onset.emotional_word.strip()
    ctx = (onset.preceding_context or "").strip()
    if ctx:
        anchor = f"{ctx} {word}"
        i = response.find(anchor)
        if i != -1:
            return i + len(ctx) + 1
        i = response.find(ctx)
        if i != -1:
            return i + len(ctx) + 1
    i = response.find(word)
    return i if i != -1 else None
