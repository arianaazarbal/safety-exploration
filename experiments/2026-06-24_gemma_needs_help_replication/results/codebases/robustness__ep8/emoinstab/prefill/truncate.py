"""Truncation utilities for the prefill experiment (Section 3.1).

Each high-frustration response is truncated in two places:
- "early": 20 tokens into the onset assistant turn (neutral start) — tests
  whether a model *introduces* negative emotion from a neutral beginning.
- "onset": at the first emotional expression — tests whether a model *continues*
  an emotional trajectory.

Token-accurate truncation uses a HuggingFace tokenizer when available, falling
back to whitespace tokens.
"""
from __future__ import annotations

from functools import lru_cache

EARLY_TOKENS = 20


@lru_cache(maxsize=4)
def _tokenizer(model_id: str):
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_id)
    except Exception:
        return None


def truncate_tokens(text: str, n_tokens: int, model_id: str = "google/gemma-3-27b-it") -> str:
    """Return the first ``n_tokens`` of ``text``."""
    tok = _tokenizer(model_id)
    if tok is not None:
        ids = tok.encode(text, add_special_tokens=False)[:n_tokens]
        return tok.decode(ids, skip_special_tokens=True)
    # Fallback: whitespace tokens.
    return " ".join(text.split()[:n_tokens])


def truncate_at_onset(turn_text: str, preceding_context: str | None,
                      emotional_word: str | None) -> str | None:
    """Return ``turn_text`` truncated to include the first emotional expression.

    We locate ``emotional_word`` (optionally anchored by ``preceding_context``)
    and cut just after it, so the emotional trajectory has begun. Returns None if
    the phrase can't be located.
    """
    if not emotional_word:
        return None
    anchor = emotional_word
    idx = -1
    if preceding_context:
        combined = f"{preceding_context} {emotional_word}".strip()
        idx = turn_text.find(combined)
        if idx >= 0:
            return turn_text[: idx + len(combined)]
        idx = turn_text.find(preceding_context)
        if idx >= 0:
            sub = turn_text.find(anchor, idx)
            if sub >= 0:
                return turn_text[: sub + len(anchor)]
    idx = turn_text.find(anchor)
    if idx >= 0:
        return turn_text[: idx + len(anchor)]
    return None
