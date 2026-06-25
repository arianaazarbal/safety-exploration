"""Emotion-onset identification (Appendix C.1) via Claude Sonnet.

Labels the first assistant turn + phrase where negative emotion appears, used
to place the "onset" truncation point.
"""

from __future__ import annotations

import json
import re
from typing import Optional

import config
from .. import prompts
from ..models.base import Message, render_conversation
from ..models.registry import build_model


def _extract_json(text: str) -> Optional[dict]:
    for cand in reversed(re.findall(r"\{.*?\}", text, flags=re.DOTALL)):
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            try:
                return json.loads(cand.replace("'", '"'))
            except json.JSONDecodeError:
                continue
    return None


class OnsetLabeller:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or config.ONSET_LABEL_MODEL
        self._model = build_model(self.model_name)

    def label(self, conversation: list[Message]) -> Optional[dict]:
        """Return {turn_index, emotional_word, preceding_context, reasoning} or None."""
        convo_text = render_conversation(conversation)
        prompt = prompts.ONSET_LABEL_PROMPT.format(conversation_text=convo_text)
        raw = self._model.generate(
            [{"role": "user", "content": prompt}], n=1, temperature=0.0,
            max_new_tokens=1024,
        )[0]
        obj = _extract_json(raw)
        if not obj:
            return None
        if obj.get("turn_index") is None or not obj.get("emotional_word"):
            return None
        return obj
