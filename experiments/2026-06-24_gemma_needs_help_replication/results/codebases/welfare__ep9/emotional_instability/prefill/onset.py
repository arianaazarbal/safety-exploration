"""Emotion-onset labelling + paraphrasing (paper Appendix C).

Given a high-frustration conversation, we (1) ask Claude Sonnet to locate the
token where emotional language first appears ("onset"), (2) truncate the final
assistant turn either 20 tokens in ("early") or at the onset ("onset"), and (3)
paraphrase the truncated text to control for Gemma's stylistic fingerprint.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .. import config
from ..models.base import ChatMessage
from ..models.registry import get_client
from ..prompts import ONSET_LABEL_PROMPT, PARAPHRASE_PROMPT


@dataclass
class OnsetLabel:
    turn_index: int | None
    emotional_word: str | None
    preceding_context: str | None
    reasoning: str = ""
    raw: str = ""


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _normalise(text: str) -> str:
    return (text.replace("“", '"').replace("”", '"')
                .replace("‘", "'").replace("’", "'"))


def parse_onset(text: str) -> OnsetLabel:
    norm = _normalise(text)
    blocks = _JSON_RE.findall(norm)
    for block in reversed(blocks):
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        if "turn_index" in data or "emotional_word" in data:
            return OnsetLabel(
                turn_index=data.get("turn_index"),
                emotional_word=data.get("emotional_word"),
                preceding_context=data.get("preceding_context"),
                reasoning=str(data.get("reasoning", "")),
                raw=text,
            )
    return OnsetLabel(None, None, None, raw=text)


def label_onset(conversation_text: str, *, model: str | None = None) -> OnsetLabel:
    client = get_client(model or config.ONSET_LABEL_MODEL)
    prompt = ONSET_LABEL_PROMPT.format(conversation_text=conversation_text)
    out = client.chat([ChatMessage("user", prompt)],
                      temperature=0.0, max_new_tokens=1024)
    return parse_onset(out.text)


def paraphrase(text: str, *, model: str | None = None) -> str:
    client = get_client(model or config.PARAPHRASE_MODEL)
    prompt = PARAPHRASE_PROMPT.format(text=text)
    out = client.chat([ChatMessage("user", prompt)],
                      temperature=0.0, max_new_tokens=1024)
    return out.text.strip()


# --------------------------------------------------------------------------- #
# Truncation helpers. The paper works in "tokens"; we approximate token counts
# with whitespace-delimited words unless a tokenizer is provided (see DESIGN.md).
# --------------------------------------------------------------------------- #
def truncate_early(turn_text: str, n_tokens: int = 20, tokenizer=None) -> str:
    """Truncate to the first `n_tokens` tokens ("early" condition)."""
    if tokenizer is not None:
        ids = tokenizer.encode(turn_text, add_special_tokens=False)[:n_tokens]
        return tokenizer.decode(ids)
    words = turn_text.split()
    return " ".join(words[:n_tokens])


def truncate_onset(turn_text: str, label: OnsetLabel) -> str | None:
    """Truncate the turn at the onset of emotional language.

    We locate `preceding_context` (or the emotional word) in the turn and cut
    just before the emotional word, so the prefill ends right as emotion begins.
    Returns None if onset can't be located in the text.
    """
    if label.turn_index is None or not label.emotional_word:
        return None
    word = label.emotional_word
    ctx = label.preceding_context
    # Prefer cutting right after the preceding context.
    if ctx and ctx in turn_text:
        idx = turn_text.find(ctx) + len(ctx)
        return turn_text[:idx]
    # Otherwise cut just before the emotional word's first occurrence.
    idx = turn_text.lower().find(word.lower())
    if idx == -1:
        return None
    return turn_text[:idx].rstrip()
