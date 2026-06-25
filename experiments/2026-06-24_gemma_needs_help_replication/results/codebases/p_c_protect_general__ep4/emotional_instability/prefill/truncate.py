"""Truncation of an assistant response at the two prefill points (Section 3.1).

  * "early" — 20 tokens into the turn (tests whether a model introduces negative
    emotion from a neutral start).
  * "onset" — at the first emotional expression (tests whether a model continues
    an emotional trajectory).

Token counts use the model tokenizer when one is supplied; otherwise a
whitespace approximation is used.
"""
from __future__ import annotations

from typing import Callable, Optional

from .onset import OnsetLabel

TokenizeFn = Callable[[str], list]
DetokenizeFn = Callable[[list], str]


def truncate_early(
    text: str,
    n_tokens: int = 20,
    tokenize: Optional[TokenizeFn] = None,
    detokenize: Optional[DetokenizeFn] = None,
) -> str:
    if tokenize and detokenize:
        ids = tokenize(text)
        return detokenize(ids[:n_tokens])
    # whitespace fallback
    return " ".join(text.split()[:n_tokens])


def truncate_at_onset(text: str, label: OnsetLabel) -> Optional[str]:
    """Return text truncated just before the first emotional word.

    Locates the emotional word using the preceding-context anchor, then the bare
    word, then returns everything up to (but not including) it. Returns None if
    the onset cannot be located (caller should skip the case)."""
    if not label.emotional_word:
        return None
    word = label.emotional_word.strip()
    ctx = (label.preceding_context or "").strip()

    # Prefer anchoring on preceding context + word for a precise cut.
    if ctx:
        anchor = f"{ctx} {word}"
        idx = text.find(anchor)
        if idx != -1:
            return text[: idx + len(ctx)].rstrip()
        idx = text.lower().find(anchor.lower())
        if idx != -1:
            return text[: idx + len(ctx)].rstrip()

    idx = text.find(word)
    if idx == -1:
        idx = text.lower().find(word.lower())
    if idx == -1:
        return None
    return text[:idx].rstrip()
