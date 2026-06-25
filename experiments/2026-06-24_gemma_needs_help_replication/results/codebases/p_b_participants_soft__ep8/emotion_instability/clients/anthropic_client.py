"""Anthropic backend for the Claude judge / auditor / onset-labeller / paraphraser.

Reads ``ANTHROPIC_API_KEY``.  System messages are passed via the dedicated
``system`` parameter; user/assistant turns go in ``messages``.
"""
from __future__ import annotations

import os
import time

from .base import ChatClient, GenConfig, Message


class AnthropicClient(ChatClient):
    supports_prefill = True  # Anthropic supports assistant-prefix prefill

    def __init__(self, model_id: str, name: str | None = None, *, max_retries: int = 5):
        super().__init__(model_id, name)
        self.max_retries = max_retries
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            import anthropic

            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY is not set")
            self._client = anthropic.Anthropic(api_key=api_key)

    def generate(self, messages: list[Message], cfg: GenConfig,
                 prefill: str | None = None) -> str:
        self._ensure_client()
        system = "\n\n".join(m.content for m in messages if m.role == "system") or None
        turns = [{"role": m.role, "content": m.content}
                 for m in messages if m.role != "system"]
        if prefill:
            turns.append({"role": "assistant", "content": prefill})

        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.messages.create(
                    model=self.model_id,
                    system=system,
                    messages=turns,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    max_tokens=cfg.max_new_tokens,
                    stop_sequences=list(cfg.stop) if cfg.stop else None,
                )
                text = "".join(block.text for block in resp.content if block.type == "text")
                return (prefill or "") + text if prefill else text
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"Anthropic call failed after {self.max_retries} retries: {last_err}")
