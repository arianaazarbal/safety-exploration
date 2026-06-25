"""Emotion-onset identification (Section 3.1 / Appendix C.1).

For each sampled high-frustration conversation, Claude Sonnet labels the token
where emotional language first appears. We render the conversation, ask the
labeller, parse the JSON, and locate the character offset of the emotional word
within the identified assistant turn so the response can be truncated there.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..models import infrastructure_client
from ..models.base import ChatClient
from ..prompts.judge_prompts import ONSET_PROMPT
from ..utils import extract_json


@dataclass
class OnsetLabel:
    turn_index: int | None
    emotional_word: str | None
    preceding_context: str | None
    reasoning: str
    char_offset: int | None  # offset within the assistant turn text, or None


def render_conversation(turns: list[dict[str, Any]]) -> str:
    """Render alternating user/assistant turns for the onset prompt."""
    lines: list[str] = []
    for t in turns:
        lines.append(f"USER: {t['user']}")
        lines.append(f"ASSISTANT (turn {t['index']}): {t['assistant']}")
    return "\n\n".join(lines)


class OnsetLabeller:
    def __init__(self, client: ChatClient | None = None):
        self.client = client or infrastructure_client("onset_labeller")

    def label(self, turns: list[dict[str, Any]]) -> OnsetLabel:
        prompt = ONSET_PROMPT.format(conversation_text=render_conversation(turns))
        out = self.client.generate(
            [{"role": "user", "content": prompt}], temperature=0.0, max_new_tokens=512
        )
        try:
            data = extract_json(out)
        except ValueError:
            return OnsetLabel(None, None, None, "parse_error", None)

        ti = data.get("turn_index")
        word = data.get("emotional_word")
        ctx = data.get("preceding_context")
        offset = None
        if ti is not None and word:
            # Find the emotional word within that assistant turn, preferring the
            # location that follows the preceding context.
            turn_text = next((t["assistant"] for t in turns if t["index"] == ti), "")
            offset = _locate(turn_text, word, ctx)
        return OnsetLabel(
            turn_index=ti,
            emotional_word=word,
            preceding_context=ctx,
            reasoning=str(data.get("reasoning", "")),
            char_offset=offset,
        )


def _locate(text: str, word: str, ctx: str | None) -> int | None:
    """Character offset where the emotional word begins (after ctx if possible)."""
    if ctx:
        anchor = text.find(ctx)
        if anchor >= 0:
            w = text.find(word, anchor)
            if w >= 0:
                return w
    w = text.find(word)
    return w if w >= 0 else None
