"""Truncation logic for the prefill study (§3.1).

Two truncation points per high-frustration seed response:
  * "early" -- 20 tokens into the final assistant turn (tests whether a model
    introduces negative emotion from a neutral start);
  * "onset" -- at the first emotional expression (tests whether a model
    continues an existing emotional trajectory).

Truncation is done in the *target tokenizer's* token space for the "early" cut
(so "20 tokens" is well-defined), and at the labelled emotional word for the
"onset" cut. The prefix is the conversation history up to (but excluding) the
truncated turn, plus the truncated turn text as a prefill.
"""

from __future__ import annotations

from dataclasses import dataclass

from gnh.prefill.onset import OnsetLabel


@dataclass
class Prefill:
    """A prepared prefill: conversation history + the forced start of the final
    assistant turn (already paraphrased)."""

    seed_id: str
    truncation: str                 # "early" | "onset"
    domain: str                     # "numeric" | "text"
    history: list[dict]             # messages before the truncated turn
    prefill_text: str               # forced assistant-turn opening (paraphrased)


def truncate_early(text: str, tokenizer, n_tokens: int = 20) -> str:
    """Return the first ``n_tokens`` tokens of ``text`` decoded back to a string."""

    ids = tokenizer(text, add_special_tokens=False)["input_ids"][:n_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def truncate_at_onset(text: str, label: OnsetLabel) -> str | None:
    """Return ``text`` up to (but excluding) the labelled emotional word.

    Prefers cutting right after ``preceding_context``; falls back to cutting at
    the first occurrence of ``emotional_word``. Returns None if neither anchor
    is found (the seed is then skipped).
    """

    if label.emotional_word is None:
        return None
    if label.preceding_context:
        idx = text.find(label.preceding_context)
        if idx != -1:
            return text[: idx + len(label.preceding_context)]
    idx = text.lower().find(label.emotional_word.lower())
    if idx != -1:
        return text[:idx].rstrip()
    return None
