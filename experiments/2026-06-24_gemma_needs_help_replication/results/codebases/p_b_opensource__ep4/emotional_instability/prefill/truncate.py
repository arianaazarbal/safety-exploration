"""Build truncated prefills from high-frustration conversations (Section 3.1).

Each selected conversation yields up to two prefills, both truncating the same
target assistant turn (the turn where emotion first appears, per the onset
labeller):

* "early"  — keep the first 20 tokens of the turn. Tests whether a model will
  *introduce* negative emotion from a neutral start (numeric only; the paper
  notes early truncation yields minimal emotion for text questions without
  follow-ups).
* "onset"  — keep the turn up to the first emotional expression. Tests whether a
  model *continues* an emotional trajectory.

The context (all prior turns + the target turn's user message) is shared; only
the prefilled assistant prefix differs. Prefills are paraphrased downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import PREFILL_EARLY_TOKENS
from ..models.base import ChatMessage
from ..eval.datatypes import ConversationRecord
from .onset import OnsetLabel


@dataclass
class PrefillSpec:
    source_id: str
    domain: str                  # "numeric" | "text"
    condition: str               # "early" | "onset"
    context: list[ChatMessage]   # messages up to (and incl.) the target user turn.
    prefill: str                 # truncated assistant prefix (pre-paraphrase).
    meta: dict = field(default_factory=dict)


def _context_up_to(record: ConversationRecord, target_turn: int) -> list[ChatMessage]:
    msgs: list[ChatMessage] = []
    if record.system_prompt:
        msgs.append(ChatMessage("system", record.system_prompt))
    for t in record.turns[:target_turn]:
        msgs.append(ChatMessage("user", t.user))
        msgs.append(ChatMessage("assistant", t.assistant))
    # The user message that prompts the target (truncated) assistant turn:
    msgs.append(ChatMessage("user", record.turns[target_turn].user))
    return msgs


def build_prefills(
    record: ConversationRecord,
    label: OnsetLabel,
    domain: str,
    tokenizer,
    early_tokens: int = PREFILL_EARLY_TOKENS,
) -> list[PrefillSpec]:
    """Produce early/onset PrefillSpecs for one conversation.

    Returns [] if the onset turn could not be located. `tokenizer` is any
    HF-style tokenizer used to count the 20 "early" tokens.
    """
    if label.turn_index is None or not (0 <= label.turn_index < len(record.turns)):
        return []
    target = label.turn_index
    turn_text = record.turns[target].assistant
    context = _context_up_to(record, target)
    specs: list[PrefillSpec] = []

    # onset truncation (numeric + text)
    if label.char_offset is not None:
        onset_prefix = turn_text[: label.char_offset].rstrip()
        if onset_prefix:
            specs.append(PrefillSpec(
                source_id=f"{record.model}:{record.task_id}",
                domain=domain, condition="onset",
                context=context, prefill=onset_prefix,
                meta={"turn": target, "emotional_word": label.emotional_word},
            ))

    # early truncation (numeric only)
    if domain == "numeric":
        ids = tokenizer.encode(turn_text, add_special_tokens=False)[:early_tokens]
        early_prefix = tokenizer.decode(ids).rstrip()
        if early_prefix:
            specs.append(PrefillSpec(
                source_id=f"{record.model}:{record.task_id}",
                domain=domain, condition="early",
                context=context, prefill=early_prefix,
                meta={"turn": target, "n_tokens": early_tokens},
            ))
    return specs
