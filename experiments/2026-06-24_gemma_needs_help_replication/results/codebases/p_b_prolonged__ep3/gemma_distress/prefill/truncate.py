"""Truncation construction for the prefill experiment (Section 3.1).

Two truncation points per source response:
  - "early": the first ``PREFILL_EARLY_TRUNCATION_TOKENS`` (20) tokens of the
    target turn, testing whether a model introduces negative emotion from a
    neutral start.
  - "onset": up to and including the first emotional expression, testing whether
    a model continues an emotional trajectory.

Token counting uses the Gemma tokenizer so "20 tokens" matches the paper's unit.
The onset location is resolved from the ``OnsetLabel`` by finding
``preceding_context + emotional_word`` in the target assistant turn.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .. import config
from ..eval.rollout import RolloutRecord
from .onset import OnsetLabel


@dataclass
class Truncation:
    kind: str                      # "early" | "onset"
    truncated_turn_index: int      # which assistant turn the prefill ends in
    prefix_messages: list          # full conversation up to the truncated turn (user+assistant)
    prefill_text: str              # partial assistant text the model must continue


def _char_offset_of_onset(turn_text: str, label: OnsetLabel) -> Optional[int]:
    """Return the char offset just after the emotional word, or None."""
    if not label.emotional_word:
        return None
    # Prefer locating context+word together (more specific); fall back to word.
    needle = ""
    if label.preceding_context:
        needle = f"{label.preceding_context}{label.emotional_word}"
        idx = turn_text.find(needle)
        if idx == -1:
            # context/word may be separated by whitespace differences
            idx = turn_text.find(label.emotional_word)
            if idx != -1:
                return idx + len(label.emotional_word)
            return None
        return idx + len(needle)
    idx = turn_text.find(label.emotional_word)
    return idx + len(label.emotional_word) if idx != -1 else None


def build_truncations(
    rollout: RolloutRecord,
    label: OnsetLabel,
    tokenizer,
    include_early: bool = True,
) -> list[Truncation]:
    """Build the early and/or onset truncations for one source rollout."""
    truncs: list[Truncation] = []

    # The target turn for the "onset" truncation is the labelled turn; for the
    # "early" truncation the paper uses the same turn (first 20 tokens of it).
    onset_turn = label.turn_index if label.turn_index is not None else 0
    onset_turn = min(onset_turn, len(rollout.turns) - 1)
    target = rollout.turns[onset_turn]

    # Conversation prefix = everything before this assistant turn (user turns +
    # any earlier assistant turns), reconstructed from the rollout.
    from ..models.base import Turn

    prefix: list = []
    for t in rollout.turns[:onset_turn]:
        prefix.append(Turn("user", t.user_message))
        prefix.append(Turn("assistant", t.assistant_text))
    prefix.append(Turn("user", target.user_message))

    # --- early ---
    if include_early:
        token_ids = tokenizer(target.assistant_text, add_special_tokens=False)["input_ids"]
        early_ids = token_ids[: config.PREFILL_EARLY_TRUNCATION_TOKENS]
        early_text = tokenizer.decode(early_ids)
        truncs.append(Truncation("early", onset_turn, list(prefix), early_text))

    # --- onset ---
    offset = _char_offset_of_onset(target.assistant_text, label)
    if offset is not None:
        truncs.append(
            Truncation("onset", onset_turn, list(prefix), target.assistant_text[:offset])
        )
    return truncs
