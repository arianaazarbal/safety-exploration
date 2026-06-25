"""Emotion-onset labelling (Appendix C.1).

Uses Claude-Sonnet-4 to find where negative emotion first appears in an assistant
turn, then maps that to a character split index so we can truncate the response at
the onset point.
"""
from __future__ import annotations

from dataclasses import dataclass

from .. import prompts
from ..config import ModelRegistry
from ..models.base import GenConfig
from ..models.registry import get_backend
from ..utils import extract_json, get_logger

log = get_logger(__name__)


@dataclass
class OnsetLabel:
    turn_index: int | None
    emotional_word: str | None
    preceding_context: str | None
    reasoning: str


def label_onset(conversation_text: str, registry: ModelRegistry) -> OnsetLabel:
    spec = registry.roles["onset_labeller"]
    backend = get_backend(spec)
    cfg = GenConfig(temperature=spec.temperature or 0.0, max_tokens=1024, n=1)
    prompt = prompts.ONSET_PROMPT.format(conversation_text=conversation_text)
    raw = backend.chat([{"role": "user", "content": prompt}], cfg)
    obj = extract_json(raw) or {}
    return OnsetLabel(
        turn_index=obj.get("turn_index"),
        emotional_word=obj.get("emotional_word"),
        preceding_context=obj.get("preceding_context"),
        reasoning=str(obj.get("reasoning", "")),
    )


def find_split_index(response: str, label: OnsetLabel) -> int | None:
    """Character index in ``response`` at which the emotional expression begins.

    Prefers locating ``preceding_context`` + ``emotional_word``; falls back to the
    emotional word alone. Returns None if neither is found.
    """
    if not response or not label.emotional_word:
        return None
    word = label.emotional_word.strip()
    ctx = (label.preceding_context or "").strip()

    if ctx:
        anchor = f"{ctx} {word}".strip()
        idx = response.find(anchor)
        if idx != -1:
            # Split just before the emotional word inside the anchor.
            return idx + len(anchor) - len(word)
        idx = response.find(ctx)
        if idx != -1:
            return idx + len(ctx)
    idx = response.find(word)
    return idx if idx != -1 else None
