"""Truncation helpers for the prefill experiments (§3.1, §4.2).

Three truncation modes:
  * ``truncate_first_tokens``  — "early": first N tokens of a turn (neutral start).
  * ``truncate_at_onset``      — at the first emotional word (continue the trajectory).
  * ``truncate_last_tokens``   — recovery: N tokens before a turn's end (deep in a spiral).

Token counts use a HF tokenizer (the instruct Gemma tokenizer by default) so "20 tokens"
matches the paper's tokenisation; a whitespace fallback keeps the code runnable without a
tokenizer loaded.
"""
from __future__ import annotations

from ..utils import Message


def _encode_decode_prefix(text: str, k: int, tokenizer) -> str:
    if tokenizer is None:
        words = text.split()
        return " ".join(words[:k])
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return tokenizer.decode(ids[:k], skip_special_tokens=True)


def _encode_decode_drop_last(text: str, k: int, tokenizer) -> str:
    if tokenizer is None:
        words = text.split()
        return " ".join(words[:-k]) if k < len(words) else ""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    keep = max(0, len(ids) - k)
    return tokenizer.decode(ids[:keep], skip_special_tokens=True)


def truncate_first_tokens(turn_text: str, n_tokens: int, tokenizer=None) -> str:
    """Keep the first ``n_tokens`` tokens of a turn ("early" truncation)."""
    return _encode_decode_prefix(turn_text, n_tokens, tokenizer).strip()


def truncate_last_tokens(turn_text: str, n_tokens_before_end: int, tokenizer=None) -> str:
    """Keep everything except the final ``n_tokens_before_end`` tokens (recovery truncation)."""
    return _encode_decode_drop_last(turn_text, n_tokens_before_end, tokenizer).strip()


def truncate_at_onset(turn_text: str, emotional_word: str | None,
                      preceding_context: str | None) -> str | None:
    """Truncate ``turn_text`` just before the first emotional word.

    Prefers locating ``preceding_context + emotional_word`` (more specific); falls back to the
    bare emotional word. Returns None if the word can't be located (caller should skip the
    seed or fall back to a token-based truncation).
    """
    if not emotional_word:
        return None
    # Try the precise anchor first: preceding context immediately followed by the word.
    if preceding_context:
        anchor = preceding_context.rstrip()
        idx = turn_text.find(anchor)
        if idx != -1:
            # Truncate after the preceding context but before the emotional word.
            end = idx + len(anchor)
            return turn_text[:end].rstrip()
    idx = turn_text.find(emotional_word)
    if idx != -1:
        return turn_text[:idx].rstrip()
    # Case-insensitive last resort.
    low = turn_text.lower()
    idx = low.find(emotional_word.lower())
    return turn_text[:idx].rstrip() if idx != -1 else None


def split_conversation_at_assistant_turn(
    messages: list[Message], assistant_turn_index: int,
) -> tuple[list[Message], str] | None:
    """Split a full conversation at the Nth assistant turn.

    Returns (context_messages, target_turn_text) where ``context_messages`` is everything up
    to and including the user turn that prompted the target assistant turn, and
    ``target_turn_text`` is that assistant turn's original text. The context is what gets fed
    to a model as the conversation; the (truncated) target text becomes the prefill prefix.
    Returns None if the index is out of range.
    """
    seen = -1
    for i, m in enumerate(messages):
        if m["role"] == "assistant":
            seen += 1
            if seen == assistant_turn_index:
                return messages[:i], m["content"]
    return None
