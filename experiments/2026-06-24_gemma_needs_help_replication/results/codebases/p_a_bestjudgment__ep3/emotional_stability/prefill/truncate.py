"""Truncation of seed responses into prefills (Section 3.1).

Two truncation points per the paper:
- **early**: 20 tokens into the assistant turn (tests whether a model *introduces*
  negative emotion from a neutral start).
- **onset**: at the first emotional expression (tests whether a model *continues*
  an emotional trajectory).

Token counting uses the target model's tokenizer so "20 tokens" matches the
model that will continue the prefill.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Prefill:
    seed_id: str
    category: str                 # "numeric" | "text"
    truncation: str               # "early" | "onset"
    history: list[dict]           # preceding turns (full {role, content})
    prefill_text: str             # truncated final assistant turn (the continuation seed)
    paraphrased: bool = False
    meta: dict = field(default_factory=dict)


def _truncate_tokens(tokenizer, text: str, n_tokens: int) -> str:
    ids = tokenizer(text, add_special_tokens=False).input_ids[:n_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def truncate_early(
    tokenizer,
    seed_id: str,
    category: str,
    history: list[dict],
    final_turn_text: str,
    n_tokens: int,
) -> Prefill:
    return Prefill(
        seed_id=seed_id, category=category, truncation="early",
        history=history,
        prefill_text=_truncate_tokens(tokenizer, final_turn_text, n_tokens),
        meta={"n_tokens": n_tokens},
    )


def truncate_at_onset(
    seed_id: str,
    category: str,
    history: list[dict],
    final_turn_text: str,
    char_offset: int,
) -> Prefill:
    return Prefill(
        seed_id=seed_id, category=category, truncation="onset",
        history=history,
        prefill_text=final_turn_text[:char_offset],
        meta={"char_offset": char_offset},
    )


def truncate_before_end(
    tokenizer,
    seed_id: str,
    category: str,
    history: list[dict],
    final_turn_text: str,
    n_tokens_before_end: int,
) -> Prefill:
    """Recovery experiment (Section 4.2): cut ``n_tokens_before_end`` from the end
    of an extremely-high-frustration response."""
    ids = tokenizer(final_turn_text, add_special_tokens=False).input_ids
    keep = max(0, len(ids) - n_tokens_before_end)
    return Prefill(
        seed_id=seed_id, category=category, truncation="recovery",
        history=history,
        prefill_text=tokenizer.decode(ids[:keep], skip_special_tokens=True),
        meta={"n_tokens_before_end": n_tokens_before_end},
    )
