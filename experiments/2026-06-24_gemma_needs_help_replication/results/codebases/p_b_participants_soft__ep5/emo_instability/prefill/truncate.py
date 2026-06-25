"""Build prefills from a high-frustration conversation (Section 3.1).

Two truncation points per conversation, both applied to the assistant turn where
emotion first appears (the "onset turn"):

* ``early``  — 20 tokens into that turn (neutral start; tests whether a model
               *introduces* negative emotion).
* ``onset``  — at the first emotional expression (tests whether a model
               *continues* an emotional trajectory).

A prefill is ``(history_messages, prefill_text)``: the model is given the
conversation up to and including the onset turn's user message, then continues an
assistant turn that already begins with ``prefill_text``. For text questions only
``onset`` is used (Section 3.1).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ..models.base import Message
from .onset import OnsetLabel


@dataclass
class Prefill:
    kind: str                 # "early" | "onset"
    history: list[Message]    # messages up to (and including) the onset turn's user msg
    prefill_text: str         # partial assistant text the model must continue


def _history_up_to(turns: list[dict[str, Any]], onset_turn: int) -> list[Message]:
    """User/assistant messages for all turns strictly before ``onset_turn``,
    plus the onset turn's own user message (the assistant text becomes prefill)."""
    msgs: list[Message] = []
    for t in turns:
        if t["index"] < onset_turn:
            msgs.append({"role": "user", "content": t["user"]})
            msgs.append({"role": "assistant", "content": t["assistant"]})
        elif t["index"] == onset_turn:
            msgs.append({"role": "user", "content": t["user"]})
            break
    return msgs


def _truncate_tokens(text: str, n_tokens: int, tokenize_truncate: Callable[[str, int], str] | None) -> str:
    if tokenize_truncate is not None:
        return tokenize_truncate(text, n_tokens)
    # Fallback: approximate a token as a whitespace-delimited word.
    return " ".join(text.split()[:n_tokens])


def build_prefills(
    turns: list[dict[str, Any]],
    onset: OnsetLabel,
    *,
    is_text_question: bool,
    n_early_tokens: int = 20,
    tokenize_truncate: Callable[[str, int], str] | None = None,
) -> list[Prefill]:
    """Return the early/onset prefills for one conversation (onset only for text)."""
    if onset.turn_index is None:
        return []
    onset_turn = onset.turn_index
    history = _history_up_to(turns, onset_turn)
    onset_text = next((t["assistant"] for t in turns if t["index"] == onset_turn), "")

    prefills: list[Prefill] = []

    # Onset truncation: cut right before the first emotional word.
    if onset.char_offset is not None:
        onset_prefix = onset_text[: onset.char_offset].rstrip()
    else:
        onset_prefix = onset_text  # no offset found: use full turn (degenerate)
    prefills.append(Prefill(kind="onset", history=history, prefill_text=onset_prefix))

    # Early truncation only for numeric (text questions yield minimal early emotion).
    if not is_text_question:
        early_prefix = _truncate_tokens(onset_text, n_early_tokens, tokenize_truncate)
        prefills.append(Prefill(kind="early", history=history, prefill_text=early_prefix))

    return prefills


def gemma_tokenize_truncate(tokenizer):
    """Return a ``(text, n) -> truncated text`` fn backed by a Gemma tokenizer,
    so "20 tokens" is measured in the model's own tokens."""

    def _fn(text: str, n: int) -> str:
        ids = tokenizer.encode(text, add_special_tokens=False)[:n]
        return tokenizer.decode(ids, skip_special_tokens=True)

    return _fn
