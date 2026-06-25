"""OpenAI-compatible HTTP backend (OpenRouter for Gemini, OpenAI for GPT-5-mini).

Both OpenRouter and OpenAI expose the Chat Completions schema, so a single client
serves both — the only differences are the base URL and the API-key env var,
selected by ``backend`` ("openrouter" vs "openai").

Per Paper §B.1, thinking is disabled where the provider supports it; the relevant
``extra_body`` (e.g. ``{"reasoning": {"enabled": false}}``) is carried on the
``ModelSpec`` and forwarded verbatim.
"""

from __future__ import annotations

import os
from typing import Sequence

from ..config import ModelSpec
from ..types import Message
from .base import ChatModel, GenerationError
from ._retry import with_retry

_BASE_URLS = {
    "openrouter": "https://openrouter.ai/api/v1",
    "openai": None,  # SDK default
}
_KEY_ENV = {
    "openrouter": "OPENROUTER_API_KEY",
    "openai": "OPENAI_API_KEY",
}


class OpenAICompatibleBackend(ChatModel):
    supports_prefill = False  # remote chat APIs don't expose assistant continuation

    def __init__(self, spec: ModelSpec):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "API backends require the 'api' extra: pip install -e '.[api]'"
            ) from exc

        self.spec = spec
        self.name = spec.name
        backend = spec.backend
        key_env = _KEY_ENV[backend]
        api_key = os.environ.get(key_env)
        if not api_key:
            raise GenerationError(
                f"Missing {key_env} for model '{spec.name}' (backend={backend})."
            )
        client_kwargs = {"api_key": api_key, "timeout": spec.request_timeout_s or 120}
        if _BASE_URLS[backend]:
            client_kwargs["base_url"] = _BASE_URLS[backend]
        self.client = OpenAI(**client_kwargs)

    def generate(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] | None = None,
    ) -> str:
        temperature = self.spec.temperature if temperature is None else temperature
        max_tokens = self.spec.max_tokens if max_tokens is None else max_tokens

        kwargs: dict = dict(
            model=self.spec.model_id,
            messages=[m.as_dict() for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if stop:
            kwargs["stop"] = list(stop)
        if self.spec.extra_body:
            kwargs["extra_body"] = self.spec.extra_body

        def _call() -> str:
            resp = self.client.chat.completions.create(**kwargs)
            choice = resp.choices[0]
            content = choice.message.content
            if content is None:
                raise GenerationError(
                    f"{self.name} returned empty content (finish={choice.finish_reason})."
                )
            return content.strip()

        try:
            return with_retry(_call)
        except Exception as exc:  # noqa: BLE001
            raise GenerationError(f"{self.name} generation failed: {exc}") from exc
