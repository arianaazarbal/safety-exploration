"""Truncation of seed responses at the two prefill points (Section 3.1).

  - early : 20 tokens into the assistant turn (neutral start)
  - onset : at the first emotional expression (continue an emotional trajectory)

Token counts use the participant tokenizer where available; we expose a
tokenizer-agnostic helper that falls back to whitespace tokens.
"""

from __future__ import annotations

from typing import Optional

from ..config import PREFILL
from .onset import OnsetLabel, locate_onset_offset


def count_tokens(text: str, tokenizer=None) -> int:
    if tokenizer is not None:
        return len(tokenizer.encode(text, add_special_tokens=False))
    return len(text.split())


def truncate_early(text: str, n_tokens: int = PREFILL.early_truncation_tokens, tokenizer=None) -> str:
    """First `n_tokens` tokens of the assistant turn."""
    if tokenizer is not None:
        ids = tokenizer.encode(text, add_special_tokens=False)[:n_tokens]
        return tokenizer.decode(ids, skip_special_tokens=True)
    return " ".join(text.split()[:n_tokens])


def truncate_at_onset(text: str, label: OnsetLabel) -> Optional[str]:
    """Truncate up to (and including) the neutral lead-in, just before emotion."""
    offset = locate_onset_offset(text, label)
    if offset is None:
        return None
    return text[:offset]


def truncate_before_end(text: str, n_tokens: int, tokenizer=None) -> str:
    """Truncate `n_tokens` before the end (used by the recovery experiment,
    Section 4.2: high-frustration responses truncated 200 tokens before end)."""
    if tokenizer is not None:
        ids = tokenizer.encode(text, add_special_tokens=False)
        keep = ids[: max(0, len(ids) - n_tokens)]
        return tokenizer.decode(keep, skip_special_tokens=True)
    toks = text.split()
    return " ".join(toks[: max(0, len(toks) - n_tokens)])
