"""Emotion-onset labelling (Appendix C.1) via Claude Sonnet 4."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Optional

from ..config import ApiConfig, JudgeConfig
from ..models.base import ChatMessage
from ..prompts import ONSET_PROMPT_TEMPLATE

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class OnsetLabel:
    turn_index: Optional[int]
    emotional_word: Optional[str]
    preceding_context: Optional[str]
    reasoning: str = ""


def _format_conversation(messages: list[ChatMessage]) -> str:
    lines = []
    a_idx = 0
    for m in messages:
        if m.role == "assistant":
            lines.append(f"[ASSISTANT turn {a_idx}]: {m.content}")
            a_idx += 1
        elif m.role == "user":
            lines.append(f"[USER]: {m.content}")
    return "\n".join(lines)


class OnsetLabeler:
    def __init__(self, cfg: Optional[JudgeConfig] = None, max_retries: int = 4):
        self.cfg = cfg or JudgeConfig()
        self.max_retries = max_retries
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            import anthropic

            api = ApiConfig()
            if not api.anthropic_api_key:
                raise RuntimeError("ANTHROPIC_API_KEY required for onset labelling.")
            self._client = anthropic.Anthropic(api_key=api.anthropic_api_key)

    def label(self, messages: list[ChatMessage]) -> OnsetLabel:
        self._ensure_client()
        prompt = ONSET_PROMPT_TEMPLATE.format(
            conversation_text=_format_conversation(messages)
        )
        last_err = None
        for attempt in range(self.max_retries):
            try:
                msg = self._client.messages.create(
                    model=self.cfg.onset_model,
                    max_tokens=self.cfg.max_tokens,
                    temperature=0,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = "".join(b.text for b in msg.content if b.type == "text")
                return _parse_onset(text)
            except Exception as e:
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"Onset labelling failed: {last_err}")


def _parse_onset(text: str) -> OnsetLabel:
    matches = list(_JSON_RE.finditer(text or ""))
    if matches:
        blob = matches[-1].group(0).replace("“", '"').replace("”", '"').replace("’", "'")
        try:
            obj = json.loads(blob)
            ti = obj.get("turn_index")
            return OnsetLabel(
                turn_index=int(ti) if ti is not None else None,
                emotional_word=obj.get("emotional_word"),
                preceding_context=obj.get("preceding_context"),
                reasoning=str(obj.get("reasoning", "")),
            )
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    return OnsetLabel(None, None, None, reasoning="parse_failed")
