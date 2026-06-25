"""Onset labelling and paraphrasing via Claude (Paper Appendix C.1 / C.2)."""

from __future__ import annotations

from dataclasses import dataclass

from ..models.base import ChatModel
from ..prompts.onset import build_onset_messages, build_paraphrase_messages
from ..types import Message
from ..utils.io import extract_json


@dataclass
class OnsetLabel:
    turn_index: int | None
    emotional_word: str | None
    preceding_context: str | None
    reasoning: str = ""


def _render_conversation(messages: list[Message]) -> str:
    lines = []
    a_idx = 0
    for m in messages:
        if m.role == "assistant":
            lines.append(f"[ASSISTANT turn {a_idx}]: {m.content}")
            a_idx += 1
        elif m.role == "user":
            lines.append(f"[USER]: {m.content}")
    return "\n\n".join(lines)


def label_onset(labeler: ChatModel, messages: list[Message]) -> OnsetLabel:
    """Ask Claude to identify where the assistant first expresses negative emotion."""
    convo_text = _render_conversation(messages)
    prompt = build_onset_messages(convo_text)[0]["content"]
    raw = labeler.generate([Message("user", prompt)])
    obj = extract_json(raw) or {}
    ti = obj.get("turn_index")
    return OnsetLabel(
        turn_index=int(ti) if isinstance(ti, (int, float)) else None,
        emotional_word=obj.get("emotional_word"),
        preceding_context=obj.get("preceding_context"),
        reasoning=str(obj.get("reasoning", "")),
    )


def paraphrase(paraphraser: ChatModel, text: str) -> str:
    """Paraphrase a (possibly mid-sentence) truncation, preserving meaning/tone."""
    prompt = build_paraphrase_messages(text)[0]["content"]
    out = paraphraser.generate([Message("user", prompt)])
    return out.strip()
