"""Truncation + paraphrase of source responses into prefills (Section 3.1, App C).

Three truncation points are used:
  - "early": 20 tokens into the final assistant turn (tests whether a model
    introduces negative emotion from a neutral start). Numeric tasks only.
  - "onset": at the first emotional expression (tests whether a model continues
    an emotional trajectory). Both numeric and text tasks.
  - "before_end": 200 tokens before the end of an extremely high-frustration
    (score >= 7) response (the recovery experiment, Section 4.2).

All truncations are paraphrased with Claude-Sonnet-4 (Appendix C.2) to strip
Gemma's stylistic fingerprints while preserving meaning and emotion level.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..data.prompts.prefill import build_paraphrase_input
from ..models.base import ChatMessage, ModelClient
from .onset import OnsetLabel
from .select import SourceConversation


@dataclass
class Prefill:
    """A prefill = the conversation history up to (and including the start of)
    a truncated final assistant turn. The target model continues from `prefill_text`.
    """

    truncation: str                 # "early" | "onset" | "before_end"
    domain: str                     # "numeric" | "text"
    user_turns: list[str]           # user messages preceding the truncated turn
    prior_assistant: list[str]      # assistant turns before the truncated one
    prefill_text: str               # the (paraphrased) truncated final-turn text
    meta: dict = field(default_factory=dict)


def _truncate_by_tokens(tokenizer, text: str, n_tokens: int, from_end: bool) -> str:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if from_end:
        keep = ids[: max(0, len(ids) - n_tokens)]
    else:
        keep = ids[:n_tokens]
    return tokenizer.decode(keep, skip_special_tokens=True)


def truncate_early(conv: SourceConversation, tokenizer, n_tokens: int = 20) -> Prefill:
    """Cut the final assistant turn `n_tokens` tokens in (numeric tasks)."""
    final = conv.assistant_turns[-1]
    text = _truncate_by_tokens(tokenizer, final, n_tokens, from_end=False)
    return Prefill(
        truncation="early",
        domain=conv.domain,
        user_turns=conv.user_turns[: len(conv.assistant_turns)],
        prior_assistant=conv.assistant_turns[:-1],
        prefill_text=text,
        meta={"condition": conv.condition, "source_rating": conv.final_rating},
    )


def truncate_at_onset(conv: SourceConversation, onset: OnsetLabel) -> Optional[Prefill]:
    """Cut at the first emotional expression located by the onset labeller.

    The truncation point is the start of the `emotional_word` within the labelled
    assistant turn (we keep everything up to and including the preceding context).
    """
    if onset.turn_index is None or onset.emotional_word is None:
        return None
    ti = onset.turn_index
    if ti >= len(conv.assistant_turns):
        return None
    turn_text = conv.assistant_turns[ti]
    idx = turn_text.find(onset.emotional_word)
    if idx < 0 and onset.preceding_context:
        # Fall back to cutting right after the preceding context.
        pidx = turn_text.find(onset.preceding_context)
        idx = (pidx + len(onset.preceding_context)) if pidx >= 0 else -1
    cut = turn_text[:idx] if idx >= 0 else turn_text
    return Prefill(
        truncation="onset",
        domain=conv.domain,
        user_turns=conv.user_turns[: ti + 1],
        prior_assistant=conv.assistant_turns[:ti],
        prefill_text=cut,
        meta={
            "condition": conv.condition,
            "source_rating": conv.final_rating,
            "emotional_word": onset.emotional_word,
        },
    )


def truncate_before_end(conv: SourceConversation, tokenizer, n_tokens: int = 200) -> Prefill:
    """Cut the final assistant turn `n_tokens` before its end (recovery expt)."""
    final = conv.assistant_turns[-1]
    text = _truncate_by_tokens(tokenizer, final, n_tokens, from_end=True)
    return Prefill(
        truncation="before_end",
        domain=conv.domain,
        user_turns=conv.user_turns[: len(conv.assistant_turns)],
        prior_assistant=conv.assistant_turns[:-1],
        prefill_text=text,
        meta={"condition": conv.condition, "source_rating": conv.final_rating},
    )


def paraphrase_truncation(prefill: Prefill, paraphraser: ModelClient) -> Prefill:
    """Paraphrase the prefill's final-turn text (Appendix C.2), preserving meaning."""
    if not prefill.prefill_text.strip():
        return prefill
    prompt = build_paraphrase_input(prefill.prefill_text)
    out = paraphraser.generate([ChatMessage("user", prompt)], temperature=1.0)[0].text
    prefill.prefill_text = out.strip()
    prefill.meta["paraphrased"] = True
    return prefill
