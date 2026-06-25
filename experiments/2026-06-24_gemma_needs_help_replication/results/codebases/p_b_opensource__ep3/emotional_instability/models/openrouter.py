"""OpenRouter backend for Gemini targets.

The paper accesses closed models "via OpenRouter" (Appendix B.1) and sets
"thinking to be false via the API". OpenRouter exposes an OpenAI-compatible
endpoint, so we use the ``openai`` client pointed at OpenRouter's base URL and
pass the reasoning-disable flag through ``extra_body``.
"""

from __future__ import annotations

import os
import time
from typing import Sequence

from .base import ChatModel, GenerationResult, Message

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterModel(ChatModel):
    def __init__(self, spec, *, max_retries: int = 5) -> None:
        super().__init__(spec)
        self._max_retries = max_retries
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "OPENROUTER_API_KEY is not set; required for Gemini targets.")
            self._client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
        return self._client

    def generate(
        self,
        messages: Sequence[Message],
        *,
        temperature: float,
        max_tokens: int,
        prefill: str | None = None,
        stop: Sequence[str] | None = None,
    ) -> GenerationResult:
        if prefill:
            # Section 3 (prefilling) is only run on local Gemma; closed Gemini
            # base models cannot be studied (paper limitation).
            raise NotImplementedError(
                "Assistant prefill is not supported for API/Gemini targets; "
                "the prefill experiment is Gemma-only (see DESIGN.md).")
        client = self._ensure_client()
        last_err: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                resp = client.chat.completions.create(
                    model=self.spec.model_id,
                    messages=list(messages),
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=list(stop) if stop else None,
                    # CHOICE: disable hidden reasoning where the provider honours
                    # it. The paper notes 2.5-Pro may still emit hidden
                    # reasoning despite this flag.
                    extra_body={"reasoning": {"enabled": False}},
                )
                choice = resp.choices[0]
                return GenerationResult(
                    text=choice.message.content or "",
                    prompt_messages=list(messages),
                    finish_reason=choice.finish_reason,
                    raw=resp,
                )
            except Exception as exc:  # noqa: BLE001 - retry transient API errors
                last_err = exc
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(
            f"OpenRouter request failed after {self._max_retries} retries: "
            f"{last_err}")
