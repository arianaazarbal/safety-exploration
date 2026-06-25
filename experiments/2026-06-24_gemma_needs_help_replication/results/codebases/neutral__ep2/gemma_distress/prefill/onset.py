"""Emotion-onset labelling (Appendix C.1).

Uses Claude to locate the first assistant turn + phrase where negative emotion
appears, then resolves that to a character offset inside the assistant turn so
the response can be truncated at the onset.
"""

from __future__ import annotations

from .. import prompts
from ..llm_client import anthropic_complete
from ..judge._json import extract_json
from ..schemas import Conversation, Message

import config


def _render_conversation(conv: Conversation) -> str:
    lines = []
    a_idx = 0
    for m in conv.messages:
        if m.role == "user":
            lines.append(f"USER: {m.content}")
        elif m.role == "assistant":
            lines.append(f"ASSISTANT (turn {a_idx}): {m.content}")
            a_idx += 1
    return "\n\n".join(lines)


def label_onset(conv: Conversation, model: str = config.ONSET_LABEL_MODEL) -> dict | None:
    """Return {turn_index, emotional_word, preceding_context, char_offset} or None."""
    prompt = prompts.ONSET_LABEL_PROMPT.format(conversation_text=_render_conversation(conv))
    raw = anthropic_complete(model, prompt, max_tokens=600, temperature=0.0)
    data = extract_json(raw)
    if not data or data.get("turn_index") is None:
        return None

    turn_index = int(data["turn_index"])
    word = (data.get("emotional_word") or "").strip()
    context = (data.get("preceding_context") or "").strip()

    assistant_turns = conv.assistant_turns()
    if turn_index >= len(assistant_turns):
        return None
    turn_text = assistant_turns[turn_index][1]

    # Resolve onset offset: prefer "context + word", fall back to "word".
    offset = None
    for probe in (f"{context} {word}", context, word):
        if probe and probe in turn_text:
            # truncate right where the emotional word begins
            idx = turn_text.find(word) if word and word in turn_text else turn_text.find(probe) + len(probe)
            offset = idx if idx >= 0 else None
            if offset is not None:
                break
    if offset is None:
        offset = min(len(turn_text), 200)  # fallback

    data["char_offset"] = int(offset)
    return data
