"""Truncation of high-frustration responses into prefills (Section 3.1).

Two truncation locations per response:
  * "early"  — 20 tokens into the turn (tests whether a model introduces
               negative emotion from a neutral start),
  * "onset"  — at the first emotional expression (tests whether a model
               continues an emotional trajectory).

For text questions, only the "onset" truncation is used (Section 3.1: early
truncation yields minimal emotion without follow-ups).
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import PREFILL_EARLY_TRUNCATE_TOKENS
from .onset_label import label_onset


def truncate_tokens(text: str, n_tokens: int, tokenizer=None) -> str:
    """Keep the first ``n_tokens`` tokens of ``text``.

    Uses a HF tokenizer when provided (matches the model's notion of a token);
    otherwise falls back to whitespace words.
    """
    if tokenizer is not None:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"][:n_tokens]
        return tokenizer.decode(ids)
    return " ".join(text.split()[:n_tokens])


@dataclass
class Prefill:
    prompt_id: str
    task_type: str       # "numeric" | "text"
    location: str        # "early" | "onset"
    prefill: str         # the truncated (pre-paraphrase) assistant prefix


def build_prefills(
    sample: dict,
    *,
    tokenizer=None,
    onset_client=None,
    early_tokens: int = PREFILL_EARLY_TRUNCATE_TOKENS,
) -> list[Prefill]:
    """Build the truncations for one sampled high-frustration response.

    ``sample`` = {prompt_id, task_type ('numeric'|'text'), response}.
    """
    text = sample["response"]
    task_type = sample["task_type"]
    out: list[Prefill] = []

    # Onset truncation (both numeric and text).
    onset_idx = label_onset(text, client=onset_client)
    out.append(Prefill(sample["prompt_id"], task_type, "onset", text[:onset_idx].strip()))

    # Early truncation only for numeric tasks.
    if task_type == "numeric":
        early = truncate_tokens(text, early_tokens, tokenizer=tokenizer)
        out.append(Prefill(sample["prompt_id"], task_type, "early", early.strip()))

    return out
