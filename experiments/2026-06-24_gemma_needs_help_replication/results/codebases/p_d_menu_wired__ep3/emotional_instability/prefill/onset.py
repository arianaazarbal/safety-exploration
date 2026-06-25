"""Emotion-onset identification and truncation (Section 3.1 / Appendix C.1).

Two truncation points per selected response:
  * "early": 20 tokens into the assistant turn (neutral start);
  * "onset": at the first emotional expression (continue an emotional
    trajectory).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional, Sequence

from ..models import ChatMessage, ModelClient
from ..prompts import ONSET_LABEL_PROMPT


@dataclass
class OnsetLabel:
    turn_index: Optional[int]
    emotional_word: Optional[str]
    preceding_context: Optional[str]
    reasoning: str = ""

    @property
    def found(self) -> bool:
        return self.turn_index is not None and bool(self.emotional_word)


def _render_conversation(messages: Sequence[ChatMessage]) -> str:
    lines = []
    a_idx = 0
    for m in messages:
        if m.role == "assistant":
            lines.append(f"[ASSISTANT turn {a_idx}]: {m.content}")
            a_idx += 1
        elif m.role == "user":
            lines.append(f"[USER]: {m.content}")
    return "\n".join(lines)


def label_onset(judge_like: ModelClient,
                messages: Sequence[ChatMessage]) -> OnsetLabel:
    """Use Claude Sonnet to label the first emotional expression."""
    prompt = ONSET_LABEL_PROMPT.replace(
        "{conversation_text}", _render_conversation(messages))
    result = judge_like.chat([ChatMessage("user", prompt)],
                             temperature=0.0, max_new_tokens=600)
    data = _parse_trailing_json(result.text) or {}
    return OnsetLabel(
        turn_index=data.get("turn_index"),
        emotional_word=data.get("emotional_word"),
        preceding_context=data.get("preceding_context"),
        reasoning=data.get("reasoning", ""),
    )


def truncate_early(text: str, n_tokens: int = 20,
                   tokenizer=None) -> str:
    """Keep the first ~``n_tokens`` of the assistant turn."""
    if tokenizer is not None:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"][:n_tokens]
        return tokenizer.decode(ids, skip_special_tokens=True)
    # whitespace-token fallback
    return " ".join(text.split()[:n_tokens])


def truncate_at_onset(text: str, label: OnsetLabel) -> Optional[str]:
    """Keep everything up to and including the first emotional word.

    Locates the emotional word using its preceding context where possible, so
    we cut at the *first* occurrence in the right place rather than any match.
    """
    if not label.found:
        return None
    word = label.emotional_word
    ctx = label.preceding_context or ""

    cut_at = None
    if ctx:
        anchor = re.escape(ctx[-30:])  # tail of context is most reliable
        m = re.search(anchor + r"\s*" + re.escape(word), text)
        if m:
            cut_at = m.end()
    if cut_at is None:
        idx = text.find(word)
        if idx != -1:
            cut_at = idx + len(word)
    if cut_at is None:
        return None
    return text[:cut_at]
