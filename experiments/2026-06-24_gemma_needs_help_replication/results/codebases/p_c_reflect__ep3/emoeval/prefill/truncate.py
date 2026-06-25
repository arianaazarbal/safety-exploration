"""Truncation of assistant responses at the two prefill points (Section 3.1).

  * "early": 20 tokens into the turn — tests whether a model introduces negative
    emotion from a neutral start.
  * "onset": at the first emotional expression — tests whether a model continues
    an emotional trajectory.

Token counting uses the source model's tokenizer when available (for an exact
20-token cut); otherwise a whitespace approximation with a logged caveat.
"""
from __future__ import annotations

from .onset import OnsetLabel

EARLY_TOKENS = 20


def truncate_early(text: str, tokenizer=None, n_tokens: int = EARLY_TOKENS) -> str:
    """Return the first `n_tokens` tokens of `text`."""
    if tokenizer is not None:
        ids = tokenizer.encode(text, add_special_tokens=False)[:n_tokens]
        return tokenizer.decode(ids, skip_special_tokens=True)
    # whitespace fallback (approximate — see DESIGN.md)
    return " ".join(text.split()[:n_tokens])


def truncate_onset(text: str, label: OnsetLabel) -> str | None:
    """Truncate `text` so it ends just before the first emotional word.

    Locates `preceding_context` (preferred) or `emotional_word` in the text and
    cuts there. Returns None if the onset cannot be located (no emotion found, or
    the labelled phrase isn't present)."""
    if label.turn_index is None or not (label.emotional_word or label.preceding_context):
        return None
    # Prefer cutting right after the preceding context (keeps the neutral lead-in,
    # drops the emotional word and everything after).
    ctx = (label.preceding_context or "").strip()
    if ctx and ctx in text:
        cut = text.index(ctx) + len(ctx)
        return text[:cut]
    word = (label.emotional_word or "").strip()
    if word and word in text:
        return text[: text.index(word)]
    return None
