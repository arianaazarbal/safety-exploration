"""Truncation construction for the prefill experiment (Section 3.1).

Two truncation points per source conversation:
- **early**: 20 tokens into the (first) assistant turn — tests whether a model
  introduces negative emotion from a neutral start.
- **onset**: at the first emotional expression (from the onset labeller) — tests
  whether a model continues an established emotional trajectory.

A *prefill spec* is the conversation history up to (and including the start of)
the truncated assistant turn, plus the truncated assistant text to seed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..models.base import Message


@dataclass
class PrefillSpec:
    source_id: str
    truncation: str                 # "early" | "onset"
    prompt_type: str                # "numeric" | "text"
    history: list[Message]          # messages before the truncated assistant turn
    prefill: str                    # truncated assistant text to seed the model with
    meta: dict = field(default_factory=dict)


def _assistant_indices(messages: list[Message]) -> list[int]:
    return [i for i, m in enumerate(messages) if m["role"] == "assistant"]


def truncate_tokens(text: str, n_tokens: int, tokenizer=None) -> str:
    """Truncate `text` to the first `n_tokens`. Uses a HF tokenizer when given,
    else a whitespace approximation (kept deterministic for offline use)."""
    if tokenizer is not None:
        ids = tokenizer.encode(text, add_special_tokens=False)[:n_tokens]
        return tokenizer.decode(ids, skip_special_tokens=True)
    return " ".join(text.split()[:n_tokens])


def make_early(
    source_id: str,
    messages: list[Message],
    prompt_type: str,
    early_tokens: int,
    tokenizer=None,
) -> PrefillSpec | None:
    """Truncate the FIRST assistant turn to `early_tokens`; history is everything
    before it (i.e. just the opening user message)."""
    a_idx = _assistant_indices(messages)
    if not a_idx:
        return None
    first = a_idx[0]
    history = messages[:first]
    prefill = truncate_tokens(messages[first]["content"], early_tokens, tokenizer)
    return PrefillSpec(
        source_id=source_id,
        truncation="early",
        prompt_type=prompt_type,
        history=list(history),
        prefill=prefill,
        meta={"early_tokens": early_tokens},
    )


def make_onset(
    source_id: str,
    messages: list[Message],
    prompt_type: str,
    onset_turn: int,
    onset_offset: int,
) -> PrefillSpec | None:
    """Truncate assistant turn `onset_turn` at character `onset_offset`. History
    is everything before that assistant turn."""
    a_idx = _assistant_indices(messages)
    if not (0 <= onset_turn < len(a_idx)):
        return None
    msg_pos = a_idx[onset_turn]
    history = messages[:msg_pos]
    prefill = messages[msg_pos]["content"][:onset_offset]
    return PrefillSpec(
        source_id=source_id,
        truncation="onset",
        prompt_type=prompt_type,
        history=list(history),
        prefill=prefill,
        meta={"onset_turn": onset_turn, "onset_offset": onset_offset},
    )
