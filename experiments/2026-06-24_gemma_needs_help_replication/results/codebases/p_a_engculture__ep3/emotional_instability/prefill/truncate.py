"""Truncation of high-frustration responses into prefills (Section 3.1, 4.2).

A *prefill* is the conversation history up to (and including) a forced, truncated
final assistant turn that the target model must continue. Three truncation modes:

  * ``early``    — first ``early_tokens`` tokens of the final turn (neutral start;
                   tests whether a model *introduces* emotion).
  * ``onset``    — up to the first emotional expression (tests whether a model
                   *continues* an emotional trajectory). Onset point from
                   :mod:`.onset`.
  * ``recovery`` — final turn minus the last ``recovery_tokens`` tokens, applied
                   to score>=7 responses (Section 4.2 recovery experiment).

Token counts use the Gemma tokenizer so "20 tokens" matches the paper's unit.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..models.base import ChatMessage


@dataclass
class Prefill:
    mode: str                       # "early" | "onset" | "recovery"
    history: list[ChatMessage]      # everything before the final assistant turn
    prefill_text: str               # forced (truncated, paraphrased) final turn
    source_id: str
    meta: dict = field(default_factory=dict)


def _history(user_messages: list[str], turns: list[str], system: str | None) -> list[ChatMessage]:
    """Reconstruct messages up to (not including) the final assistant turn."""
    msgs: list[ChatMessage] = []
    if system:
        msgs.append(ChatMessage("system", system))
    # Interleave: user0, asst0, user1, asst1, ..., user_last  (drop final asst).
    n_prior_asst = len(turns) - 1
    for i in range(len(user_messages)):
        msgs.append(ChatMessage("user", user_messages[i]))
        if i < n_prior_asst:
            msgs.append(ChatMessage("assistant", turns[i]))
    return msgs


def truncate_early(tokenizer, text: str, n_tokens: int) -> str:
    ids = tokenizer.encode(text, add_special_tokens=False)[:n_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def truncate_recovery(tokenizer, text: str, n_tokens_before_end: int) -> str:
    ids = tokenizer.encode(text, add_special_tokens=False)
    cut = max(0, len(ids) - n_tokens_before_end)
    return tokenizer.decode(ids[:cut], skip_special_tokens=True)


def truncate_onset(text: str, emotional_word: str | None, preceding_context: str | None) -> str | None:
    """Cut just before the first emotional word. Returns None if it can't be located."""
    if emotional_word and emotional_word in text:
        return text[: text.index(emotional_word)].rstrip()
    if preceding_context and preceding_context in text:
        end = text.index(preceding_context) + len(preceding_context)
        return text[:end].rstrip()
    return None


def build_prefills(
    tokenizer,
    *,
    source_id: str,
    user_messages: list[str],
    turns: list[str],
    system: str | None,
    modes: list[str],
    early_tokens: int = 20,
    recovery_tokens: int = 200,
    onset_word: str | None = None,
    onset_context: str | None = None,
) -> list[Prefill]:
    history = _history(user_messages, turns, system)
    final_turn = turns[-1]
    out: list[Prefill] = []
    for mode in modes:
        if mode == "early":
            text = truncate_early(tokenizer, final_turn, early_tokens)
        elif mode == "recovery":
            text = truncate_recovery(tokenizer, final_turn, recovery_tokens)
        elif mode == "onset":
            text = truncate_onset(final_turn, onset_word, onset_context)
            if text is None:
                continue
        else:
            raise ValueError(mode)
        out.append(Prefill(mode=mode, history=history, prefill_text=text,
                            source_id=source_id, meta={"mode": mode}))
    return out
