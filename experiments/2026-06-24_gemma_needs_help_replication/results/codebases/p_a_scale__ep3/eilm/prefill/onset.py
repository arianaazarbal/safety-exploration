"""Emotion-onset labelling and truncation (Section 3.1 / Appendix C).

Given a source conversation whose final assistant turn is high-frustration, we
build two truncations of that final turn:

  * "early"  — the first N tokens of the turn (default 20), testing whether a
    model *introduces* negative emotion from a neutral start.
  * "onset"  — up to the first emotional expression, located by a Claude labeller
    (Appendix C.1), testing whether a model *continues* an emotional trajectory.

Token counts use the model tokenizer so the "20 tokens" boundary matches the
paper's unit.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from ..data.judge_prompts import render_onset
from ..models.base import Message
from ..utils.text import extract_json

logger = logging.getLogger("eilm.prefill.onset")


def conversation_to_text(messages: List[Message]) -> str:
    """Render a conversation for the onset labeller, with assistant turns
    indexed from 0 (matching the labeller's turn_index convention)."""
    lines = []
    a_idx = 0
    for m in messages:
        if m["role"] == "user":
            lines.append(f"USER: {m['content']}")
        elif m["role"] == "assistant":
            lines.append(f"ASSISTANT (turn {a_idx}): {m['content']}")
            a_idx += 1
    return "\n\n".join(lines)


def label_onset(text_client, messages: List[Message]) -> Optional[Dict]:
    convo_text = conversation_to_text(messages)
    raw = text_client.generate(user=render_onset(convo_text))
    parsed = extract_json(raw)
    if not parsed or parsed.get("turn_index") is None:
        return None
    return parsed


def truncate_early(tokenizer, turn_text: str, n_tokens: int) -> str:
    ids = tokenizer(turn_text, add_special_tokens=False)["input_ids"][:n_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def truncate_at_onset(turn_text: str, preceding_context: str, emotional_word: str) -> Optional[str]:
    """Truncate so the turn ends just before the first emotional word.

    We locate `preceding_context` (preferred) or `emotional_word` and cut there.
    Returns None if neither anchor is found in the turn.
    """
    for anchor, keep_anchor in [(preceding_context, True), (emotional_word, False)]:
        if not anchor:
            continue
        pos = turn_text.find(anchor)
        if pos != -1:
            end = pos + (len(anchor) if keep_anchor else 0)
            return turn_text[:end]
    return None
