"""Anthropic API client.

Used purely as *infrastructure*: the frustration judge (Claude Sonnet 4), the
emotion-onset labeller and paraphraser (Claude Sonnet 4), the Petri auditor
(Claude Sonnet 4) and the Petri judge (Claude Opus 4). These are NOT eval
targets in this replication.
"""
from __future__ import annotations

import time
from typing import Optional

from .. import config
from .base import ChatMessage, GenerationResult, ModelClient


class AnthropicClient(ModelClient):
    def __init__(self, name: str, model_id: str, *, max_retries: int = 5):
        super().__init__(name)
        self.model_id = model_id
        self.max_retries = max_retries
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic  # noqa: WPS433

            if not config.ANTHROPIC_API_KEY:
                raise RuntimeError("ANTHROPIC_API_KEY is not set.")
            self._client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        return self._client

    def _chat(self, messages, *, temperature, max_new_tokens, prefill, stop):
        client = self._get_client()

        system = None
        api_messages = []
        for m in messages:
            if m.role == "system":
                system = (system + "\n\n" + m.content) if system else m.content
            else:
                api_messages.append({"role": m.role, "content": m.content})
        if prefill:
            api_messages.append({"role": "assistant", "content": prefill})

        kwargs = dict(
            model=self.model_id,
            max_tokens=max_new_tokens,
            temperature=temperature,
            messages=api_messages,
        )
        if system:
            kwargs["system"] = system
        if stop:
            kwargs["stop_sequences"] = stop

        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = client.messages.create(**kwargs)
                content = "".join(
                    block.text for block in resp.content
                    if getattr(block, "type", None) == "text"
                )
                full = (prefill or "") + content
                return GenerationResult(
                    text=full, prefill=prefill or "",
                    finish_reason=resp.stop_reason or "",
                    raw={"id": resp.id},
                )
            except Exception as exc:  # noqa: BLE001 - broad retry
                last_err = exc
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"Anthropic request failed after {self.max_retries} retries: {last_err}")
