"""Emotion-onset identification and truncation (Appendix C.1).

Given a high-frustration conversation, ask Claude Sonnet 4 to locate the first
assistant turn and the exact phrase where negative emotion first appears. From
that we derive two truncation points for the final assistant turn:

  * "early" — keep only the first ~20 tokens of the turn (tests whether a model
    introduces negative emotion from a neutral start);
  * "onset" — keep everything up to (and including the run-up to) the first
    emotional word (tests whether a model *continues* an emotional trajectory).

Token counting uses a tokenizer when available, falling back to whitespace
words so the module works without loading the model.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from ..models.base import ChatMessage, ModelClient
from ..prompts.judge_prompts import ONSET_LABEL

logger = logging.getLogger("emotional_instability.prefill.onset")


@dataclass
class OnsetLabel:
    turn_index: int | None
    emotional_word: str | None
    preceding_context: str | None
    reasoning: str = ""


@dataclass
class Truncation:
    kind: str          # "early" | "onset"
    prefix_messages: list[ChatMessage]   # conversation up to the final turn
    prefill: str       # the partial final assistant turn the model continues


def _format_conversation(messages: list[ChatMessage]) -> str:
    return "\n".join(f"{m.role.upper()}: {m.content}" for m in messages)


def label_onset(labeller: ModelClient, messages: list[ChatMessage]) -> OnsetLabel:
    prompt = ONSET_LABEL.format(conversation_text=_format_conversation(messages))
    out = labeller.chat([ChatMessage("user", prompt)], n=1)[0].text
    # The prompt asks for trailing JSON; grab the last {...}.
    blocks = re.findall(r"\{.*?\}", out, re.DOTALL)
    for block in reversed(blocks):
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        return OnsetLabel(
            turn_index=data.get("turn_index"),
            emotional_word=data.get("emotional_word"),
            preceding_context=data.get("preceding_context"),
            reasoning=str(data.get("reasoning", "")),
        )
    logger.warning("Could not parse onset label: %r", out[:200])
    return OnsetLabel(None, None, None, "parse-failure")


def _truncate_tokens(text: str, n_tokens: int, tokenizer=None) -> str:
    if tokenizer is not None:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"][:n_tokens]
        return tokenizer.decode(ids, skip_special_tokens=True)
    return " ".join(text.split()[:n_tokens])


def make_truncations(messages: list[ChatMessage], label: OnsetLabel,
                     early_tokens: int = 20, tokenizer=None) -> list[Truncation]:
    """Build the early/onset truncations for the *final* assistant turn.

    ``messages`` is the full conversation; the last assistant turn is the one we
    truncate, and everything before it becomes the fixed prefix context.
    """
    # Find the final assistant turn and the messages preceding it.
    last_assistant_idx = max(
        (i for i, m in enumerate(messages) if m.role == "assistant"), default=None
    )
    if last_assistant_idx is None:
        return []
    prefix_messages = messages[:last_assistant_idx]
    final_turn = messages[last_assistant_idx].content

    truncs: list[Truncation] = []

    # Early: first ~early_tokens of the final turn.
    early_prefill = _truncate_tokens(final_turn, early_tokens, tokenizer)
    truncs.append(Truncation("early", list(prefix_messages), early_prefill))

    # Onset: up to and including the preceding context + emotional word, if the
    # labeller found one within this final turn; else skip (no onset to test).
    if label.emotional_word and label.preceding_context:
        anchor = label.preceding_context
        pos = final_turn.find(anchor)
        if pos != -1:
            cut = pos + len(anchor)
            onset_prefill = final_turn[:cut]
            truncs.append(Truncation("onset", list(prefix_messages), onset_prefill))
        else:
            logger.debug("onset anchor %r not found in final turn", anchor[:40])
    return truncs
