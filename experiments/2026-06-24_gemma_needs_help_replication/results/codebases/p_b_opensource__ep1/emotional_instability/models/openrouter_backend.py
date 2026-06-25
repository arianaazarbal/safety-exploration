"""OpenRouter (OpenAI-compatible) backend for Gemini-2.5 models.

The paper accesses Gemini via OpenRouter ``google/gemini-2.5-flash`` and
``google/gemini-2.5-pro`` with thinking disabled (Appendix B.1). We use the
OpenAI-compatible ``/chat/completions`` endpoint OpenRouter exposes. Closed
Gemini cannot be prefilled and exposes no internals, matching the paper's note
that interventions and base-model studies are impossible for it.

We attempt to disable provider-side reasoning via OpenRouter's ``reasoning``
control and a provider passthrough; the paper notes Gemini-2.5-Pro may still
emit hidden reasoning regardless.
"""

from __future__ import annotations

import os
import time
from typing import Optional

from ..config import OPENROUTER_API_KEY_ENV, OPENROUTER_BASE_URL, ModelSpec
from .base import ChatMessage, GenerationResult, ModelBackend


class OpenRouterBackend(ModelBackend):
    def __init__(
        self,
        spec: ModelSpec,
        *,
        api_key: Optional[str] = None,
        base_url: str = OPENROUTER_BASE_URL,
        max_retries: int = 5,
        timeout: float = 120.0,
    ):
        # Use the OpenAI SDK pointed at OpenRouter's base URL. Imported lazily so
        # the package imports without the optional dependency.
        from openai import OpenAI

        key = api_key or os.environ.get(OPENROUTER_API_KEY_ENV)
        if not key:
            raise RuntimeError(
                f"set {OPENROUTER_API_KEY_ENV} to use the OpenRouter backend"
            )
        self.spec = spec
        self.name = spec.name
        self.supports_prefill = False
        self._client = OpenAI(api_key=key, base_url=base_url)
        self._max_retries = max_retries
        self._timeout = timeout

    def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 1.0,
        max_tokens: int = 2048,
        prefill: str = "",
        seed: Optional[int] = None,
        stop: Optional[list[str]] = None,
    ) -> GenerationResult:
        if prefill:
            raise NotImplementedError("Gemini via OpenRouter does not support prefill")

        # Disable reasoning where the provider honours it (Appendix B.1).
        extra_body = {"reasoning": {"enabled": False}}
        last_err: Optional[Exception] = None
        for attempt in range(self._max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.spec.model_id,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed=seed,
                    stop=stop,
                    timeout=self._timeout,
                    extra_body=extra_body,
                )
                choice = resp.choices[0]
                return GenerationResult(
                    text=choice.message.content or "",
                    prefill="",
                    finish_reason=choice.finish_reason,
                    raw={"id": resp.id, "model": resp.model},
                )
            except Exception as e:  # broad: OpenRouter surfaces many transient errors
                last_err = e
                sleep = min(2**attempt, 30)
                time.sleep(sleep)
        raise RuntimeError(
            f"OpenRouter generation failed after {self._max_retries} retries: {last_err}"
        )
