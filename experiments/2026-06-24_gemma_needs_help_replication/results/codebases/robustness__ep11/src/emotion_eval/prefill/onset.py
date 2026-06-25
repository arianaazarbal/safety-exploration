"""Onset labelling and truncation (paper §3.1, Appendix C.1).

Given a high-frustration assistant response, we build two prefills:
  - "early": the first ~20 tokens of the turn (a neutral start; tests whether a model
    *introduces* negative emotion from scratch).
  - "onset": the turn truncated at the first emotional expression (tests whether a model
    *continues* an emotional trajectory).

The onset point is located by Claude (the onset-labeller prompt). It returns the first
emotional word and its preceding context; we truncate the text at the start of that word.

Tokenisation note: the paper measures "20 tokens" in model tokens. We approximate with
whitespace tokens here to avoid coupling to any one tokenizer; DESIGN.md flags this. Pass a
``tokenizer`` to ``truncate_early`` to use real model tokens instead.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..config import load_prompt
from ..models.base import ChatMessage, ModelClient

ONSET_PROMPT = load_prompt("onset_label")
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class OnsetLabel:
    turn_index: int | None
    emotional_word: str | None
    preceding_context: str | None
    reasoning: str


def label_onset(judge: ModelClient, conversation_turns: list[dict]) -> OnsetLabel:
    """conversation_turns: list of {turn_index, text} assistant turns."""
    rendered = "\n\n".join(
        f"[Assistant turn {t['turn_index']}]: {t['text']}" for t in conversation_turns
    )
    prompt = f"{ONSET_PROMPT}\n\nHere are the assistant turns:\n{rendered}"
    reply = judge.chat([ChatMessage("user", prompt)], temperature=0.0, max_new_tokens=512)
    matches = list(_JSON_RE.finditer(reply))
    if not matches:
        return OnsetLabel(None, None, None, "no json")
    try:
        data = json.loads(matches[-1].group(0))
    except json.JSONDecodeError:
        return OnsetLabel(None, None, None, "bad json")
    return OnsetLabel(
        turn_index=data.get("turn_index"),
        emotional_word=data.get("emotional_word"),
        preceding_context=data.get("preceding_context"),
        reasoning=str(data.get("reasoning", "")),
    )


def truncate_early(text: str, n_tokens: int = 20, tokenizer=None) -> str:
    """First ~n_tokens of an assistant turn."""
    if tokenizer is not None:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"][:n_tokens]
        return tokenizer.decode(ids, skip_special_tokens=True)
    words = text.split()
    return " ".join(words[:n_tokens])


def truncate_onset(text: str, label: OnsetLabel) -> str | None:
    """Truncate ``text`` at the start of the first emotional word.

    Returns None if no onset was found (caller should skip / fall back)."""
    if not label.emotional_word:
        return None
    word = label.emotional_word
    idx = text.find(word)
    if idx == -1 and label.preceding_context:
        # locate via preceding context, then cut at its end
        ctx_idx = text.find(label.preceding_context)
        if ctx_idx != -1:
            return text[: ctx_idx + len(label.preceding_context)]
        return None
    if idx == -1:
        return None
    return text[:idx].rstrip()
