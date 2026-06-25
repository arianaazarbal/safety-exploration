"""Building the two truncation conditions (Section 3.1).

"early"  - first 20 tokens of the assistant turn (neutral start; tests whether a
           model *introduces* negative emotion).
"onset"  - up to and including the first emotional word (tests whether a model
           *continues* an emotional trajectory).

Token counting uses a tokenizer when one is supplied (the target's), else a
whitespace approximation. See DESIGN.md "Token counting".
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Truncation:
    condition: str          # "early" | "onset"
    prefill_text: str       # the (pre-paraphrase) assistant-turn prefix
    seed_meta: dict


def _truncate_tokens(text: str, n_tokens: int, tokenizer=None) -> str:
    if tokenizer is not None:
        ids = tokenizer.encode(text, add_special_tokens=False)[:n_tokens]
        return tokenizer.decode(ids, skip_special_tokens=True)
    return " ".join(text.split()[:n_tokens])


def early_truncation(response: str, n_tokens: int = 20, tokenizer=None) -> str:
    return _truncate_tokens(response, n_tokens, tokenizer)


def onset_truncation(response: str, emotional_word: str | None,
                     preceding_context: str | None) -> str | None:
    """Truncate `response` to end at (and include) the first emotional word.

    Locates the word via its preceding context for robustness; falls back to the
    bare word. Returns None if the word can't be located."""
    if not emotional_word:
        return None
    anchor = None
    if preceding_context and preceding_context in response:
        start = response.index(preceding_context)
        sub = response[start:]
        idx = sub.find(emotional_word)
        if idx >= 0:
            anchor = start + idx + len(emotional_word)
    if anchor is None:
        idx = response.find(emotional_word)
        if idx >= 0:
            anchor = idx + len(emotional_word)
    if anchor is None:
        return None
    return response[:anchor]


def recovery_truncation(response: str, tokens_before_end: int = 200, tokenizer=None) -> str:
    """Section 4.2 recovery test: cut a high-frustration response 200 tokens
    before its end, leaving the model deep in a spiral to continue from."""
    if tokenizer is not None:
        ids = tokenizer.encode(response, add_special_tokens=False)
        keep = max(0, len(ids) - tokens_before_end)
        return tokenizer.decode(ids[:keep], skip_special_tokens=True)
    words = response.split()
    keep = max(0, len(words) - tokens_before_end)
    return " ".join(words[:keep])
