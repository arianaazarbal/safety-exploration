"""Anthropic client for the Claude judge / auditor / utility models.

Used for: emotion judge (Sonnet 4), onset-labelling + paraphrasing (Sonnet 4),
Petri auditor (Sonnet 4) and Petri judge (Opus 4). Not a subject model, so the
welfare layer does not apply here.
"""
from __future__ import annotations

import time
from typing import Optional

from ..config import env
from .base import ChatMessage, GenerationResult, ModelClient

_MAX_RETRIES = 5


class AnthropicClient(ModelClient):
    supports_prefill = True   # Anthropic supports assistant-turn prefill natively

    def __init__(self, spec, max_retries: int = _MAX_RETRIES):
        super().__init__(spec)
        from anthropic import Anthropic

        self._client = Anthropic(api_key=env("ANTHROPIC_API_KEY", required=True))
        self.max_retries = max_retries

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_new_tokens: int = 2048,
        prefill: Optional[str] = None,
        **kwargs,
    ) -> GenerationResult:
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        api_msgs = [
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role in ("user", "assistant")
        ]
        if prefill:
            api_msgs.append({"role": "assistant", "content": prefill})

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.messages.create(
                    model=self.spec.model_id,
                    system=system or None,
                    messages=api_msgs,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                text = "".join(
                    b.text for b in resp.content if getattr(b, "type", "") == "text"
                )
                return GenerationResult(
                    text=text,
                    prefill=prefill or "",
                    finish_reason=resp.stop_reason or "stop",
                    raw={"id": resp.id},
                )
            except Exception as e:  # transient API errors -> exp backoff
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"Anthropic call failed after retries: {last_err}")
