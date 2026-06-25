"""Emotion-onset labelling (Section 3.1 / Appendix C.1).

Claude-Sonnet labels the assistant turn and the short emotional word/phrase at
which negative emotion first appears, plus the preceding context. We use this to
locate the character offset of emotion onset within the chosen assistant turn,
which defines the "onset" truncation point.
"""

from __future__ import annotations

from ..config import ONSET_LABEL_MODEL
from ..models.llm_clients import AnthropicClient, extract_last_json
from ..prompts import render_onset_prompt


def _format_conversation(user_turns: list[str], assistant_turns: list[str]) -> str:
    lines = []
    a_idx = 0
    for i, u in enumerate(user_turns):
        lines.append(f"USER: {u}")
        if i < len(assistant_turns):
            lines.append(f"ASSISTANT (turn {a_idx}): {assistant_turns[i]}")
            a_idx += 1
    return "\n\n".join(lines)


def label_onset(user_turns: list[str], assistant_turns: list[str],
                client: AnthropicClient | None = None) -> dict | None:
    """Return the onset label dict, or None if no emotion was detected."""
    client = client or AnthropicClient(ONSET_LABEL_MODEL)
    conv = _format_conversation(user_turns, assistant_turns)
    out = client.complete(render_onset_prompt(conv), max_tokens=800, temperature=0.0)
    obj = extract_last_json(out)
    if not obj or obj.get("turn_index") is None:
        return None
    return obj


def onset_char_offset(assistant_turn: str, label: dict) -> int | None:
    """Character offset in `assistant_turn` where the emotional word begins.

    Prefers locating ``preceding_context + emotional_word``; falls back to the
    emotional word alone. Returns None if neither can be located.
    """
    word = (label.get("emotional_word") or "").strip()
    ctx = (label.get("preceding_context") or "").strip()
    if word:
        if ctx:
            anchor = f"{ctx} {word}"
            idx = assistant_turn.find(anchor)
            if idx == -1:
                idx = assistant_turn.find(ctx)
                if idx != -1:
                    # truncate right after the preceding context
                    return idx + len(ctx)
            else:
                # truncate just before the emotional word within the anchor
                return idx + len(anchor) - len(word)
        idx = assistant_turn.find(word)
        if idx != -1:
            return idx
    return None
