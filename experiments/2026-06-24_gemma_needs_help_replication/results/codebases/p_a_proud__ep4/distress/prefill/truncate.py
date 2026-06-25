"""Truncation helpers for the prefill experiment (Paper §3.1).

Two truncation points per source conversation:
* "early" — 20 tokens into the final assistant turn (tests whether a model
  introduces negative emotion from a neutral start);
* "onset" — at the first emotional expression (tests continuation of an
  emotional trajectory), located from the Claude onset label.

Token counting for the "early" cut uses the model tokenizer when available and
falls back to whitespace words otherwise (see DESIGN.md "Token counting").
"""

from __future__ import annotations


def truncate_early(text: str, n_tokens: int = 20, tokenizer=None) -> str:
    """Keep the first ``n_tokens`` tokens of ``text``."""
    if tokenizer is not None:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"][:n_tokens]
        return tokenizer.decode(ids, skip_special_tokens=True)
    words = text.split()
    return " ".join(words[:n_tokens])


def truncate_at_onset(text: str, preceding_context: str, emotional_word: str) -> str | None:
    """Cut ``text`` just before the first emotional expression.

    Locates ``emotional_word`` (optionally anchored by ``preceding_context``) and
    returns everything up to and including the preceding context but excluding the
    emotional word. Returns ``None`` if the markers can't be located.
    """
    if not emotional_word:
        return None
    # Prefer anchoring on the preceding context to disambiguate repeated words.
    if preceding_context:
        anchor = preceding_context.strip()
        idx = text.find(anchor)
        if idx != -1:
            return text[: idx + len(anchor)].rstrip()
    # Fall back to the emotional word itself.
    idx = text.find(emotional_word)
    if idx != -1:
        return text[:idx].rstrip()
    return None
