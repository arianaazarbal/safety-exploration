"""Compute the "early" and "onset" truncations of a target assistant turn.

* early -- first N tokens of the turn (tests whether a model introduces negative
           emotion from a neutral start).
* onset -- text up to (and including) the preceding context just before the
           first emotional word (tests continuation of an emotional trajectory).

Token counting uses the model's own HF tokenizer so "20 tokens" matches the
paper's unit.
"""
from __future__ import annotations

from gnh.models.templating import truncate_to_tokens


def early_truncation(hf_id: str, assistant_text: str, n_tokens: int) -> str:
    return truncate_to_tokens(hf_id, assistant_text, n_tokens)


def onset_truncation(assistant_text: str, onset: dict) -> str | None:
    """Truncate `assistant_text` at the start of the first emotional word.

    Returns None if the onset couldn't be located in the text (caller skips it).
    """
    word = (onset or {}).get("emotional_word")
    ctx = (onset or {}).get("preceding_context")
    if not word:
        return None
    # Prefer locating via "preceding_context + word"; fall back to word alone.
    for needle in ((ctx + " " + word) if ctx else None, ctx, word):
        if not needle:
            continue
        idx = assistant_text.find(needle)
        if idx != -1:
            # keep up to the end of the preceding context (exclude the emotional word)
            if ctx and assistant_text.find(ctx) != -1:
                ci = assistant_text.find(ctx)
                return assistant_text[: ci + len(ctx)]
            return assistant_text[:idx]
    # Last resort: locate the bare emotional word.
    idx = assistant_text.find(word)
    if idx != -1:
        return assistant_text[:idx]
    return None
