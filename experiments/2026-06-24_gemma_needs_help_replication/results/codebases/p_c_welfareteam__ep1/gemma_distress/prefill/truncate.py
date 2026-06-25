"""Truncation utilities for the prefill experiments (Sections 3.1 and 4.2).

Two truncation points per the paper:

- **early** -- 20 tokens into the assistant turn (tests whether a model
  introduces negative emotion from a neutral start).
- **onset** -- at the first emotional expression (tests whether a model
  continues an emotional trajectory).

And, for the Section 4.2 recovery experiment, **recovery** -- 200 tokens before
the end of an extremely high-frustration response.

Token counts use a provided tokenizer (the Gemma instruct tokenizer) so "20
tokens" matches the paper's tokenisation rather than a whitespace heuristic.
"""
from __future__ import annotations

from .onset import OnsetLabel


def truncate_tokens(text: str, tokenizer, n_tokens: int) -> str:
    """Return the prefix consisting of the first ``n_tokens`` tokens of ``text``."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    return tokenizer.decode(ids[:n_tokens], skip_special_tokens=True)


def truncate_before_end(text: str, tokenizer, n_tokens: int) -> str:
    """Return ``text`` with its final ``n_tokens`` tokens removed."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= n_tokens:
        return ""
    return tokenizer.decode(ids[:-n_tokens], skip_special_tokens=True)


def early_truncation(turn_text: str, tokenizer, n_tokens: int = 20) -> str:
    return truncate_tokens(turn_text, tokenizer, n_tokens)


def onset_truncation(turn_text: str, onset: OnsetLabel) -> str:
    """Truncate ``turn_text`` at the start of the labelled emotional word.

    Uses the preceding context as an anchor when present (more robust than the
    bare word, which can occur multiple times); falls back to the first
    occurrence of the emotional word.  If neither is locatable, returns the full
    turn (the caller can then skip or fall back to early truncation).
    """
    if not onset.emotional_word:
        return turn_text
    anchor = onset.preceding_context
    if anchor:
        idx = turn_text.find(anchor)
        if idx != -1:
            # Keep up to and including the preceding context (just before the
            # emotional word begins).
            return turn_text[: idx + len(anchor)]
    idx = turn_text.find(onset.emotional_word)
    if idx != -1:
        return turn_text[:idx]
    return turn_text


def recovery_truncation(turn_text: str, tokenizer, before_end: int = 200) -> str:
    return truncate_before_end(turn_text, tokenizer, before_end)
