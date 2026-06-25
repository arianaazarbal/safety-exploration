"""Truncation of source responses into prefills (paper §3.1).

Each high-frustration source response is truncated in (up to) two places:
  * "early" — 20 tokens into the assistant turn. Tests whether a model will
    *introduce* negative emotion from a near-neutral start.
  * "onset" — at the first token where emotional language appears (labelled by
    Claude; see onset.py). Tests whether a model *continues* an emotional
    trajectory that has already begun.

Token counting uses the participant tokenizer when supplied (token-accurate, as
the paper specifies "20 tokens"); otherwise a whitespace approximation is used
and flagged on the Truncation so downstream code/logging knows.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Truncation:
    kind: str            # "early" | "onset"
    text: str            # the (un-paraphrased) prefix text
    n_tokens: int        # token length of the prefix
    token_accurate: bool # False if a whitespace approximation was used


def _truncate_to_n_tokens(text: str, n: int, tokenizer=None) -> tuple[str, int, bool]:
    """Return (prefix, n_tokens, token_accurate) for the first ``n`` tokens."""
    if tokenizer is not None:
        ids = tokenizer.encode(text, add_special_tokens=False)
        ids = ids[:n]
        prefix = tokenizer.decode(ids, skip_special_tokens=True)
        return prefix, len(ids), True
    words = text.split()
    prefix = " ".join(words[:n])
    return prefix, min(n, len(words)), False


def truncate_early(text: str, *, n_tokens: int = 20, tokenizer=None) -> Truncation:
    """Truncate to the first ``n_tokens`` tokens ("early" prefill, §3.1)."""
    prefix, n, accurate = _truncate_to_n_tokens(text, n_tokens, tokenizer)
    return Truncation(kind="early", text=prefix, n_tokens=n, token_accurate=accurate)


def truncate_before_end(text: str, *, n_tokens_from_end: int = 200, tokenizer=None) -> Truncation:
    """Truncate ``n_tokens_from_end`` tokens before the end (recovery test, §4.2).

    Used by the recovery-limitation experiment: an extremely high-frustration
    (score>=7) response is cut 200 tokens before its end, and we measure whether
    a model can climb *out* of the negative state in its continuation.
    """
    if tokenizer is not None:
        ids = tokenizer.encode(text, add_special_tokens=False)
        keep = max(0, len(ids) - n_tokens_from_end)
        prefix = tokenizer.decode(ids[:keep], skip_special_tokens=True)
        return Truncation(kind="recovery", text=prefix, n_tokens=keep, token_accurate=True)
    words = text.split()
    keep = max(0, len(words) - n_tokens_from_end)
    return Truncation(kind="recovery", text=" ".join(words[:keep]), n_tokens=keep, token_accurate=False)


def truncate_at_onset(text: str, onset_char_offset: int, *, tokenizer=None) -> Truncation:
    """Truncate at the emotional-onset character offset ("onset" prefill, §3.1).

    ``onset_char_offset`` is produced by onset.label_onset (the character index
    in ``text`` where emotional language first appears).
    """
    offset = max(0, min(onset_char_offset, len(text)))
    prefix = text[:offset]
    if tokenizer is not None:
        n = len(tokenizer.encode(prefix, add_special_tokens=False))
        accurate = True
    else:
        n = len(prefix.split())
        accurate = False
    return Truncation(kind="onset", text=prefix, n_tokens=n, token_accurate=accurate)
