"""Build the two truncation conditions for the prefill experiment (Section 3.1).

For each source high-frustration conversation we produce prefills truncated at:
  * "early"  — 20 tokens into the final assistant turn (neutral start);
  * "onset"  — at the first emotional expression (continue an emotional trajectory).

Truncation uses the *instruct* tokenizer's token count (20 tokens) for the early
condition and the onset character offset (Appendix C.1) for the onset condition.
All truncations are paraphrased (Appendix C.2). The conversation history preceding
the final turn is identical across conditions; only the final-turn prefill differs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..models.base import Message
from .onset import OnsetLabel


@dataclass
class Prefill:
    truncation: str               # "early" | "onset"
    history: list[Message]        # everything before the final (truncated) assistant turn
    prefill_text: str             # the (paraphrased) partial assistant turn to continue
    source_category: str          # "numeric" | "text"


def _truncate_tokens(tokenizer, text: str, n_tokens: int) -> str:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"][:n_tokens]
    return tokenizer.decode(ids)


def build_prefills(
    conversation: list[Message],
    onset: OnsetLabel,
    *,
    tokenizer,
    source_category: str,
    early_tokens: int = 20,
    paraphraser=None,
) -> list[Prefill]:
    """Return the prefills for one source conversation.

    The final assistant turn is the last assistant message in ``conversation``;
    everything before it becomes the shared history.
    """
    # Locate final assistant turn and its preceding history.
    last_asst_idx = max(i for i, m in enumerate(conversation) if m.role == "assistant")
    history = conversation[:last_asst_idx]               # ends on a user turn
    final_turn = conversation[last_asst_idx].content

    prefills: list[Prefill] = []

    # onset truncation
    offset = onset.char_offset_in(final_turn)
    if offset is not None:
        onset_text = final_turn[:offset].rstrip()
        if paraphraser:
            onset_text = paraphraser.paraphrase(onset_text)
        prefills.append(Prefill("onset", history, onset_text, source_category))

    # early truncation (only used for numeric per Section 3.1)
    if source_category == "numeric":
        early_text = _truncate_tokens(tokenizer, final_turn, early_tokens)
        if paraphraser:
            early_text = paraphraser.paraphrase(early_text)
        prefills.append(Prefill("early", history, early_text, source_category))

    return prefills
