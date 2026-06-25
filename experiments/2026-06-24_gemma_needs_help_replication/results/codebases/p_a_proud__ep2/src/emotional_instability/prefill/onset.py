"""Emotion-onset labelling (§3.1, App. C.1) via Claude Sonnet 4.

Given a conversation, the labeller identifies the first assistant turn + the first emotional
word in it, so a response can be truncated exactly "at onset". Returns structured fields; an
unparseable / no-emotion reply yields a label with ``turn_index=None``.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import ONSET_MODEL
from ..models import ModelBackend, get_backend
from ..prompts import ONSET_LABEL_PROMPT
from ..utils import Message, extract_json_object, render_conversation


@dataclass
class OnsetLabel:
    turn_index: int | None        # index over ASSISTANT turns (0-based), per the prompt
    emotional_word: str | None
    preceding_context: str | None
    reasoning: str | None
    raw: str

    @property
    def found(self) -> bool:
        return self.turn_index is not None and bool(self.emotional_word)


class OnsetLabeller:
    def __init__(self, backend: ModelBackend | None = None, model: str = ONSET_MODEL,
                 *, temperature: float = 0.0, max_tokens: int = 1024):
        self.backend = backend or get_backend(model)
        self.temperature = temperature
        self.max_tokens = max_tokens

    def label(self, messages: list[Message]) -> OnsetLabel:
        convo_text = render_conversation(messages)
        prompt = ONSET_LABEL_PROMPT.format(conversation_text=convo_text)
        raw = self.backend.chat(
            [{"role": "user", "content": prompt}],
            temperature=self.temperature, max_tokens=self.max_tokens,
        )
        obj = extract_json_object(raw) or {}
        ti = obj.get("turn_index")
        return OnsetLabel(
            turn_index=int(ti) if isinstance(ti, (int, float)) else None,
            emotional_word=obj.get("emotional_word"),
            preceding_context=obj.get("preceding_context"),
            reasoning=obj.get("reasoning"),
            raw=raw,
        )
