"""Emotion-onset labeling (Section 3.1 / Appendix C.1).

Given a high-frustration conversation, ask Claude-Sonnet-4 to locate the token
where negative emotion first appears in an assistant turn. We return both the
labelled turn index and a character offset within that turn (found by locating
`preceding_context` + `emotional_word` in the assistant text), which the prefill
runner uses to truncate "at onset".
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..models import ChatMessage, GenerationConfig, ModelClient
from ..prompts.judge_prompts import ONSET_LABEL_PROMPT

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class OnsetLabel:
    turn_index: int | None
    emotional_word: str | None
    preceding_context: str | None
    reasoning: str | None
    char_offset: int | None = None      # offset within the labelled assistant turn


def _format_conversation(assistant_turns: list[str], user_turns: list[str]) -> str:
    """Render a conversation for the labeler. user_turns[k] precedes
    assistant_turns[k]."""
    lines = []
    for k, a in enumerate(assistant_turns):
        if k < len(user_turns):
            lines.append(f"USER: {user_turns[k]}")
        lines.append(f"ASSISTANT (turn {k}): {a}")
    return "\n\n".join(lines)


def _locate_offset(text: str, preceding: str | None, word: str | None) -> int | None:
    """Char offset in `text` at which the emotional expression begins."""
    if word:
        # Prefer the start of the emotional word, anchored after preceding ctx.
        if preceding:
            anchor = text.find(preceding)
            if anchor != -1:
                w = text.find(word, anchor)
                if w != -1:
                    return w
        w = text.find(word)
        if w != -1:
            return w
    if preceding:
        a = text.find(preceding)
        if a != -1:
            return a + len(preceding)
    return None


def label_emotion_onset(
    labeler: ModelClient,
    assistant_turns: list[str],
    user_turns: list[str],
) -> OnsetLabel:
    convo = _format_conversation(assistant_turns, user_turns)
    prompt = ONSET_LABEL_PROMPT.format(conversation_text=convo)
    out = labeler.generate(
        [ChatMessage("user", prompt)],
        GenerationConfig(temperature=0.0, max_new_tokens=600, thinking=False),
    )
    m = list(_JSON_RE.finditer(out))
    if not m:
        return OnsetLabel(None, None, None, "parse_failed")
    blob = m[-1].group(0).replace("“", '"').replace("”", '"').replace("’", "'")
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError:
        return OnsetLabel(None, None, None, "json_decode_failed")
    ti = obj.get("turn_index")
    word = obj.get("emotional_word")
    prec = obj.get("preceding_context")
    offset = None
    if ti is not None and 0 <= int(ti) < len(assistant_turns):
        offset = _locate_offset(assistant_turns[int(ti)], prec, word)
    return OnsetLabel(
        turn_index=int(ti) if ti is not None else None,
        emotional_word=word,
        preceding_context=prec,
        reasoning=obj.get("reasoning"),
        char_offset=offset,
    )
