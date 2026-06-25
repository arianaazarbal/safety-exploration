"""OpenRouter-backed chat model (used for Gemini, and optionally the API judge).

OpenRouter exposes an OpenAI-compatible Chat Completions API, so we use the
``openai`` SDK pointed at the OpenRouter base URL. Requires OPENROUTER_API_KEY.
"""

from __future__ import annotations

import os
import time

from ..config import ModelSpec
from .base import GenerationResult, Message

_BASE_URL = "https://openrouter.ai/api/v1"
_MAX_RETRIES = 5


class OpenRouterChatModel:
    def __init__(self, spec: ModelSpec, api_key: str | None = None):
        self.spec = spec
        self.key = spec.key
        self.supports_prefill = False  # closed API: cannot force-continue a turn
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "openai SDK required for OpenRouter models: pip install openai"
                ) from e
            if not self._api_key:
                raise RuntimeError("OPENROUTER_API_KEY is not set")
            self._client = OpenAI(base_url=_BASE_URL, api_key=self._api_key)
        return self._client

    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float = 1.0,
        max_tokens: int = 2048,
        stop: list[str] | None = None,
    ) -> GenerationResult:
        client = self._ensure_client()
        # `extra` carries provider knobs such as reasoning.enabled=False, which
        # disables hidden "thinking" where the provider supports it.
        extra_body = dict(self.spec.extra)
        last_err: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                resp = client.chat.completions.create(
                    model=self.spec.model_id,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=stop,
                    extra_body=extra_body or None,
                )
                choice = resp.choices[0]
                return GenerationResult(
                    text=choice.message.content or "",
                    finish_reason=choice.finish_reason,
                    raw=resp.model_dump() if hasattr(resp, "model_dump") else None,
                )
            except Exception as e:  # noqa: BLE001 - broad retry on transient API errs
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(
            f"OpenRouter generation failed after {_MAX_RETRIES} retries"
        ) from last_err

    def prefill(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError(
            f"{self.spec.display_name} is a closed API model and cannot be "
            "prefilled. Prefill experiments are local-model only."
        )
