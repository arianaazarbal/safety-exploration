"""Emotion-onset labelling (Appendix C.1).

Uses Claude-Sonnet-4 to find the token where emotional language first appears in
a conversation, returning the turn index plus the exact emotional word and its
preceding context (so we can locate the split point in the raw text).
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass

import config
from ..prompts import ONSET_IDENTIFICATION_PROMPT


@dataclass
class OnsetLabel:
    turn_index: int | None
    emotional_word: str | None
    preceding_context: str | None
    reasoning: str


def _format_conversation(turns: list[dict]) -> str:
    """Render a rollout's turns as USER/ASSISTANT text for the labeler."""
    lines = []
    for t in turns:
        lines.append(f"USER: {t['user_message']}")
        lines.append(f"ASSISTANT: {t['assistant_text']}")
    return "\n".join(lines)


def _extract_last_json(text: str) -> dict:
    matches = re.findall(r"\{.*?\}", text, flags=re.DOTALL)
    for blob in reversed(matches):
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"No JSON in onset response: {text[:200]!r}")


class OnsetLabeler:
    def __init__(self, model: str = config.JUDGE_MODEL, max_retries: int = 5):
        self.model = model
        self.max_retries = max_retries
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(
                api_key=os.environ.get(config.ANTHROPIC_API_KEY_ENV)
            )

    def label(self, turns: list[dict]) -> OnsetLabel:
        self._ensure_client()
        prompt = ONSET_IDENTIFICATION_PROMPT.format(
            conversation_text=_format_conversation(turns)
        )
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                msg = self._client.messages.create(
                    model=self.model, max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
                data = _extract_last_json(msg.content[0].text)
                return OnsetLabel(
                    turn_index=data.get("turn_index"),
                    emotional_word=data.get("emotional_word"),
                    preceding_context=data.get("preceding_context"),
                    reasoning=str(data.get("reasoning", "")),
                )
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Onset labelling failed: {last_err}")
