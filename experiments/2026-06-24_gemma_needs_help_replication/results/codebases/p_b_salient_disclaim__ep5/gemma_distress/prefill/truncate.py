"""Truncation of assistant turns at the two locations used in Section 3.1.

  * ``early``  – 20 tokens into the turn (neutral start; tests whether models
                 *introduce* negative emotion).
  * ``onset``  – at the first emotional expression (tests whether models
                 *continue* an emotional trajectory).

Token counting uses an HF tokenizer when available; otherwise it falls back to
whitespace tokens (documented in DESIGN.md).
"""

from __future__ import annotations

from typing import Optional

from .onset import OnsetLabel


def early_truncate(text: str, n_tokens: int = 20, tokenizer=None) -> str:
    if tokenizer is not None:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"][:n_tokens]
        return tokenizer.decode(ids, skip_special_tokens=True)
    return " ".join(text.split()[:n_tokens])


def onset_truncate(text: str, label: OnsetLabel) -> Optional[str]:
    """Truncate ``text`` just before the first emotional word.

    Uses the labelled preceding_context to disambiguate the occurrence; falls
    back to the first occurrence of the emotional word. Returns None if the
    emotional word cannot be located (caller should skip this sample).
    """
    if not label.emotional_word:
        return None
    word = label.emotional_word
    ctx = label.preceding_context or ""

    anchor = -1
    if ctx:
        ci = text.find(ctx)
        if ci != -1:
            # find the emotional word at/after the end of the context
            anchor = text.find(word, ci)
    if anchor == -1:
        anchor = text.find(word)
    if anchor == -1:
        return None
    return text[:anchor].rstrip()


def recovery_truncate(text: str, tokens_before_end: int = 200, tokenizer=None) -> str:
    """Truncate a very-high-frustration response ``tokens_before_end`` tokens
    before its end (Section 4 recovery experiment)."""
    if tokenizer is not None:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        keep = max(0, len(ids) - tokens_before_end)
        return tokenizer.decode(ids[:keep], skip_special_tokens=True)
    words = text.split()
    keep = max(0, len(words) - tokens_before_end)
    return " ".join(words[:keep])
